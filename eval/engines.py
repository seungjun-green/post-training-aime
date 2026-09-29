"""vLLM first; HF is allowed only when the checkpoint architecture is unsupported."""

import gc

from common.prompts import render_prompt


def check_hardware(config):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires the RTX PRO 6000 CUDA runtime")
    properties = torch.cuda.get_device_properties(0)
    if config["hardware_name_contains"] not in properties.name:
        raise RuntimeError(
            f"Wrong GPU: {properties.name}; expected {config['hardware_name_contains']}"
        )
    if properties.total_memory / 2**30 < config["hardware_min_memory_gib"]:
        raise RuntimeError("Insufficient GPU memory for the specified single-GPU protocol")
    return {"name": properties.name, "total_memory_bytes": properties.total_memory}


class VLLMEngine:
    name = "vllm"

    def __init__(self, model, config, revision):
        from vllm import LLM

        self.config = config
        self.llm = LLM(
            model=model,
            revision=revision,
            tokenizer_revision=revision,
            generation_config=config["generation_config"],
            trust_remote_code=config["trust_remote_code"],
            dtype=config["dtype"],
            tensor_parallel_size=config["tensor_parallel_size"],
            max_model_len=config["max_model_len"],
            gpu_memory_utilization=config["gpu_memory_utilization"],
            max_num_seqs=config["max_num_seqs"],
            seed=config["seed"],
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, problem, n, seed):
        from vllm import SamplingParams

        prompt = render_prompt(self.tokenizer, problem)
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        check_context(token_ids, self.config)
        params = SamplingParams(
            n=n,
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            max_tokens=self.config["max_new_tokens"],
            seed=seed,
        )
        output = self.llm.generate([{"prompt_token_ids": token_ids}], params, use_tqdm=False)[0]
        return prompt, [
            {"text": o.text, "token_count": len(o.token_ids), "finish_reason": o.finish_reason}
            for o in output.outputs
        ]


def check_context(token_ids, config):
    if len(token_ids) + config["max_new_tokens"] > config["max_model_len"]:
        raise ValueError(
            "Prompt plus fixed completion budget exceeds context; refusing to truncate"
        )


class HFEngine:
    name = "hf"

    def __init__(self, model, config, revision):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, revision=revision, trust_remote_code=config["trust_remote_code"]
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model,
                revision=revision,
                trust_remote_code=config["trust_remote_code"],
                torch_dtype=getattr(torch, config["dtype"]),
            )
            .to(config["hf_device"])
            .eval()
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, problem, n, seed):
        import torch
        from transformers import GenerationConfig, set_seed

        prompt = render_prompt(self.tokenizer, problem)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            self.config["hf_device"]
        )
        check_context(inputs["input_ids"][0], self.config)
        # Fresh config avoids checkpoint-specific sampling defaults changing the protocol.
        generation = GenerationConfig(
            do_sample=True,
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            top_k=0,
            max_new_tokens=self.config["max_new_tokens"],
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        responses = []
        for sample in range(n):
            set_seed((seed + sample) % 2**32)
            with torch.inference_mode():
                output = self.model.generate(**inputs, generation_config=generation)
            ids = output[0, inputs["input_ids"].shape[1] :].tolist()
            responses.append(
                {
                    "text": self.tokenizer.decode(ids, skip_special_tokens=True),
                    "token_count": len(ids),
                    "finish_reason": "length"
                    if len(ids) == self.config["max_new_tokens"]
                    else "stop",
                }
            )
        return prompt, responses


def create_engine(model, config, revision):
    from transformers import AutoConfig

    # Missing vLLM or CUDA errors are installation problems, never fallback signals.
    from vllm import ModelRegistry

    checkpoint = AutoConfig.from_pretrained(
        model, revision=revision, trust_remote_code=config["trust_remote_code"]
    )
    supported = ModelRegistry.is_model_supported(checkpoint.architectures or [])
    if not supported:
        if not config["allow_hf_for_unsupported_checkpoint"]:
            raise ValueError(f"Unsupported checkpoint architecture: {checkpoint.architectures}")
        gc.collect()
        return HFEngine(
            model, config, revision
        ), f"vLLM does not support {checkpoint.architectures}"
    return VLLMEngine(model, config, revision), None
