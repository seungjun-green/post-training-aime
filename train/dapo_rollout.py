"""Single-GPU vLLM rollouts with sleep and explicit, audited weight synchronization."""

from pathlib import Path


class DAPOWeightSyncWorker:
    """vLLM worker extension; RPC sends a method name and path, never a callable."""

    def load_dapo_snapshot(self, path):
        from safetensors.torch import load_file
        loaded = self.get_model().load_weights(load_file(path, device="cpu").items())
        return len(loaded) if loaded is not None else None


class VLLMRollout:
    def __init__(self, source, config, work_dir):
        from vllm import LLM
        self.config = config
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.llm = LLM(model=str(source), tokenizer=str(source), trust_remote_code=True,
                       generation_config="vllm", seed=config["training"]["seed"],
                       worker_extension_cls="train.dapo_rollout.DAPOWeightSyncWorker",
                       logprobs_mode="processed_logprobs", **config["rollout"])
        self.llm.sleep(level=1)
        self.awake = False

    def sync(self, model):
        import torch
        from safetensors.torch import save_file
        # Snapshot lives on local SSD, never Drive. Only one BF16 copy is retained.
        # Explicit file transport also works when vLLM runs its engine in a child process.
        path = self.work_dir / "rollout_weights.safetensors"
        temporary = path.with_suffix(".tmp")
        weights = {name: p.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().clone()
                   for name, p in model.named_parameters()}
        save_file(weights, str(temporary))
        del weights
        temporary.replace(path)
        torch.cuda.empty_cache()
        self.llm.wake_up()
        self.awake = True
        loaded = self.llm.collective_rpc("load_dapo_snapshot", args=(str(path),))
        if any(count == 0 for count in loaded):
            raise RuntimeError("vLLM did not load the updated policy weights")
        self.llm.reset_prefix_cache()

    def generate(self, questions, step, batch_index):
        from vllm import SamplingParams
        a = self.config["algorithm"]
        # Independent response seeds are reproducible across checkpoint resume.
        prompts, params = [], []
        for qindex, question in enumerate(questions):
            for sample in range(a["group_size"]):
                from common.io import digest
                seed = int(digest([self.config["training"]["seed"], step, batch_index, question["id"], sample])[:8], 16)
                prompts.append({"prompt_token_ids": question["prompt_token_ids"]})
                params.append(SamplingParams(n=1, temperature=a["temperature"], top_p=a["top_p"],
                                             top_k=-1, max_tokens=a["max_completion_length"],
                                             logprobs=0, seed=seed))
        outputs = self.llm.generate(prompts, sampling_params=params, use_tqdm=True)
        if len(outputs) != len(prompts):
            raise ValueError("Missing vLLM requests")
        flat = []
        for expected, output in zip(prompts, outputs, strict=True):
            if output.prompt_token_ids != expected["prompt_token_ids"]:
                raise ValueError("vLLM changed or truncated the shared English prompt")
            response = output.outputs[0]
            if response.finish_reason not in {"stop", "length"}:
                raise ValueError(f"Unexpected rollout termination: {response.finish_reason}")
            flat.append({"token_ids": list(response.token_ids), "text": response.text,
                         "finish_reason": response.finish_reason,
                         "logprobs": [float(lp[token].logprob) for token, lp in
                                      zip(response.token_ids, response.logprobs, strict=True)]})
        return [flat[i:i + a["group_size"]] for i in range(0, len(flat), a["group_size"])]

    def sleep(self):
        if self.awake:
            self.llm.sleep(level=1)
            self.awake = False

    def close(self):
        self.sleep()
        # The engine core owns the vLLM worker process. Ensure it terminates on errors too.
        core = getattr(self.llm.llm_engine, "engine_core", None)
        if core is not None and hasattr(core, "shutdown"):
            core.shutdown()
