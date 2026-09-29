"""English generation adapters; reuse the existing model loaders and context guard."""

import gc

from common.english_prompts import render_prompt
from eval.engines import HFEngine, VLLMEngine, check_context


class EnglishVLLMEngine(VLLMEngine):
    def generate(self, problem, n, seed):
        from vllm import SamplingParams

        prompt = render_prompt(self.tokenizer, problem)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        check_context(ids, self.config)
        params = SamplingParams(
            n=n,
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            max_tokens=self.config["max_new_tokens"],
            seed=seed,
        )
        output = self.llm.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0]
        return prompt, [
            {"text": o.text, "token_count": len(o.token_ids), "finish_reason": o.finish_reason}
            for o in output.outputs
        ]


class EnglishHFEngine(HFEngine):
    def generate(self, problem, n, seed):
        import torch
        from transformers import GenerationConfig, set_seed

        prompt = render_prompt(self.tokenizer, problem)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            self.config["hf_device"]
        )
        check_context(inputs["input_ids"][0], self.config)
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
    from vllm import ModelRegistry

    checkpoint = AutoConfig.from_pretrained(
        model, revision=revision, trust_remote_code=config["trust_remote_code"]
    )
    if not ModelRegistry.is_model_supported(checkpoint.architectures or []):
        if not config["allow_hf_for_unsupported_checkpoint"]:
            raise ValueError(f"Unsupported checkpoint architecture: {checkpoint.architectures}")
        gc.collect()
        return EnglishHFEngine(
            model, config, revision
        ), f"Unsupported architecture: {checkpoint.architectures}"
    return EnglishVLLMEngine(model, config, revision), None
