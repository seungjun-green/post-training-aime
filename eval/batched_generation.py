"""Continuous multi-problem scheduling using the pinned vLLM 0.14.1 engine API."""

import time

from common.english_prompts import render_prompt
from eval.engines import check_context


def generate_problems(engine, jobs, execution):
    """Yield (job, prompt, responses) as whole problems finish, possibly out of order.

    jobs carry key/problem/n/seed. Native n-way sampling preserves vLLM's child
    seed derivation. FINAL_ONLY avoids transferring partial text on every token.
    More pending problems do not increase the configured GPU max_num_seqs.
    """
    if "budget_forcing" in engine.config:
        from eval.budget_generation import generate_budget_problems

        yield from generate_budget_problems(engine, jobs, execution)
        return
    if engine.name != "vllm":
        raise ValueError("Continuous evaluation requires vLLM; no silent sequential fallback")
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    core = engine.llm.llm_engine
    jobs = iter(jobs)
    pending = {}
    exhausted = False
    counter = 0
    last_status = time.monotonic()
    try:
        while pending or not exhausted:
            while not exhausted and len(pending) < execution["max_pending_problems"]:
                job = next(jobs, None)
                if job is None:
                    exhausted = True
                    break
                prompt = render_prompt(engine.tokenizer, job["problem"])
                ids = engine.tokenizer.encode(prompt, add_special_tokens=False)
                check_context(ids, engine.config)
                params = SamplingParams(
                    n=job["n"], temperature=engine.config["temperature"],
                    top_p=engine.config["top_p"], max_tokens=engine.config["max_new_tokens"],
                    seed=job["seed"], output_kind=RequestOutputKind.FINAL_ONLY,
                    **({"stop_token_ids": engine.config["stop_token_ids"], "ignore_eos": False}
                       if "stop_token_ids" in engine.config else {}),
                )
                request_id = f"eval-{counter}"
                counter += 1
                core.add_request(request_id, {"prompt_token_ids": ids}, params)
                pending[request_id] = (job, prompt)
            if not pending:
                break
            for output in core.step():
                if not output.finished:
                    continue
                job, prompt = pending.pop(output.request_id)
                outputs = sorted(output.outputs, key=lambda o: o.index)
                if [o.index for o in outputs] != list(range(job["n"])):
                    raise ValueError("vLLM returned missing or duplicate sample indices")
                yield job, prompt, [
                    {"text": o.text, "token_count": len(o.token_ids),
                     "finish_reason": o.finish_reason,
                     **({"stop_reason": getattr(o, "stop_reason", None),
                         "last_token_id": o.token_ids[-1] if o.token_ids else None}
                        if engine.config.get("record_stop_diagnostics") else {})}
                    for o in outputs
                ]
            now = time.monotonic()
            if now - last_status >= execution["status_interval_seconds"]:
                print(f"Generation active: {len(pending)} problems pending; "
                      f"GPU response limit {engine.config['max_num_seqs']}", flush=True)
                last_status = now
    finally:
        if pending:
            core.abort_request(list(pending))
