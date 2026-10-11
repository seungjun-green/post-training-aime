# Bundled transfer-evaluation sources

The upstream code and example files below are bundled verbatim, including their notices.
Exact download URLs and SHA-256 hashes are in `transfer_sources.json`.

- **IFEval**: Google Research `instruction_following_eval`, commit
  `5b381db4139c3cbf25cc9a698c5fe9754215fae2`. Apache-2.0; license in
  `instruction_following_eval/LICENSE`. Four upstream modules implement the official
  strict and loose prompt/instruction checks. The local wrapper fixes random seeds,
  points NLTK at the bundled English tables, and passes the entire original response.
- **BBH**: Mirac Suzgun et al., `BIG-Bench-Hard`, commit
  `9ee07bd481feebf959a6b59d61ea57bdcf30964d`. MIT; license in `bbh/LICENSE`.
  Nineteen original three-shot CoT demonstration files. Canary notices are retained
  in the source files; the prompt builder drops the header before `-----`, as in
  the upstream prompt format. These are evaluation demonstrations, not training data.
- **MMLU-Pro**: TIGER-AI-Lab, commit `f418b116db00b065c2aea046518d8fcf74d39872`.
  Apache-2.0; license in `mmlu_pro/LICENSE`. Original initial prompt. The local prompt
  builder follows `evaluate_from_local.py`: five validation examples of the same
  subject, in dataset order, then the test question. Test CoT/gold answers never
  enter the prompt. Our deterministic answer extractor requires an explicit final
  declaration or answer-only response, and does not randomly guess on parse failure.
- **NLTK punkt_tab English tables**: `nltk/nltk_data`, commit
  `550b6625bcef1f2abff2ff770a5a0d272c9c6b2a`. Bundled for offline official IFEval
  sentence counting. Source ZIP/member paths are recorded in the manifest.

Pinned dataset revisions, filenames, hashes and row counts are in
`configs/qwen_transfer_eval.yaml`. ARC and BBH test data use the public Hugging Face
parquet releases; IFEval uses Google's original JSONL. No benchmark data is added to
training. All four benchmarks are generative, wrapped in each model's saved chat
template; these internal comparison scores are not claimed to reproduce completion-
formatted leaderboards or ARC likelihood-based `acc_norm`.
