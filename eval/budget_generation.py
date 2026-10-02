"""Two-phase, continuously batched reasoning/answer generation for single samples."""

import time

from common.english_prompts import render_prompt
from eval.engines import check_context


def validate_budget(config):
    budget = config["budget_forcing"]
    if set(budget) != {"min_reasoning_tokens", "max_reasoning_tokens", "answer_budget_tokens",
                       "end_thinking", "answer_prefix"}:
        raise ValueError("Unexpected budget-forcing settings")
    if type(budget["min_reasoning_tokens"]) is not int or budget["min_reasoning_tokens"] != 0:
        raise ValueError("Only minimum zero is supported; no forced thinking extension")
    for key in ["max_reasoning_tokens", "answer_budget_tokens"]:
        if type(budget[key]) is not int or budget[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if budget["max_reasoning_tokens"] + budget["answer_budget_tokens"] > config["max_new_tokens"]:
        raise ValueError("Reasoning plus answer budgets exceed the total completion budget")
    if any(spec["n"] != 1 for spec in config["datasets"].values()):
        raise ValueError("Budget forcing currently supports one answer per problem")
    for key in ["end_thinking", "answer_prefix"]:
        if not isinstance(budget[key], str) or not budget[key].strip():
            raise ValueError(f"{key} must be nonempty text")


def generate_budget_problems(engine, jobs, execution):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    if engine.name != "vllm":
        raise ValueError("Budget forcing requires vLLM")
    validate_budget(engine.config)
    config, tokenizer, core = engine.config, engine.tokenizer, engine.llm.llm_engine
    budget = config["budget_forcing"]
    stop = budget["end_thinking"]
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    if len(encode("\n" + stop + budget["answer_prefix"])) >= budget["answer_budget_tokens"]:
        raise ValueError("Answer budget is too small for the delimiter/cue and at least one answer token")
    jobs, pending = iter(jobs), {}
    exhausted, counter, last_status = False, 0, time.monotonic()

    def submit(request_id, ids, state, limit, **kwargs):
        if limit < 1 or len(ids) + limit > config["max_model_len"]:
            raise ValueError("No answer room within the configured context/budget")
        params = SamplingParams(n=1, temperature=config["temperature"], top_p=config["top_p"],
                                max_tokens=limit, seed=state["job"]["seed"],
                                output_kind=RequestOutputKind.FINAL_ONLY, **kwargs)
        pending[request_id] = state
        core.add_request(request_id, {"prompt_token_ids": ids}, params)

    try:
        while pending or not exhausted:
            while not exhausted and len(pending) < execution["max_pending_problems"]:
                job = next(jobs, None)
                if job is None:
                    exhausted = True
                    break
                if job["n"] != 1:
                    raise ValueError("Budget forcing requires n=1")
                prompt = render_prompt(tokenizer, job["problem"])
                ids = encode(prompt)
                check_context(ids, config)
                state = {"job": job, "prompt": prompt, "prompt_ids": ids, "phase": "reasoning"}
                submit(f"budget-{counter}", ids, state, budget["max_reasoning_tokens"],
                       stop=[stop], include_stop_str_in_output=True)
                counter += 1
            if not pending:
                break
            for output in core.step():
                if not output.finished:
                    continue
                state = pending.pop(output.request_id)
                if len(output.outputs) != 1 or output.outputs[0].index != 0:
                    raise ValueError("Budget forcing requires exactly one completed sample")
                response = output.outputs[0]
                count = len(response.token_ids)
                if state["phase"] == "reasoning":
                    if count > budget["max_reasoning_tokens"]:
                        raise ValueError("Reasoning exceeded its configured budget")
                    natural_end = getattr(response, "stop_reason", None) == stop or response.text.endswith(stop)
                    forced = response.finish_reason == "length" and not natural_end
                    details = {"forced": forced, "reasoning_tokens": count,
                               "reasoning_finish_reason": response.finish_reason,
                               "injected_tokens": 0, "answer_tokens": 0}
                    if natural_end or forced:
                        cue = ("\n" + stop if forced else "") + budget["answer_prefix"]
                        cue_ids = encode(cue)
                        # Injected delimiters count against the 4,096 answer allowance,
                        # so the full continuation can never exceed 20,480 tokens.
                        limit = budget["answer_budget_tokens"] - len(cue_ids)
                        details["injected_tokens"] = len(cue_ids)
                        state.update(phase="answer", prefix=response.text + cue, details=details,
                                     answer_limit=limit)
                        # Preserve generated token IDs exactly; do not re-tokenize the trace.
                        ids = state["prompt_ids"] + list(response.token_ids) + cue_ids
                        submit(output.request_id + "-answer", ids, state, limit)
                        continue
                    # Natural EOS before the thinking cap: keep the completed response.
                    if response.finish_reason != "stop":
                        raise ValueError(f"Unexpected reasoning termination: {response.finish_reason}")
                    text = response.text
                else:
                    if count > state["answer_limit"] or response.finish_reason not in {"stop", "length"}:
                        raise ValueError("Invalid answer completion")
                    text = state["prefix"] + response.text
                    details = state["details"]
                    details["answer_tokens"] = count
                total = sum(details[key] for key in ["reasoning_tokens", "injected_tokens", "answer_tokens"])
                if total > config["max_new_tokens"]:
                    raise ValueError("Combined completion exceeded total token budget")
                yield state["job"], state["prompt"], [{
                    "text": text, "token_count": total, "finish_reason": response.finish_reason,
                    "budget_forcing": details,
                }]
            now = time.monotonic()
            if now - last_status >= execution["status_interval_seconds"]:
                print(f"Budget generation active: {len(pending)} problems pending; "
                      f"GPU response limit {config['max_num_seqs']}", flush=True)
                last_status = now
    finally:
        if pending:
            core.abort_request(list(pending))
