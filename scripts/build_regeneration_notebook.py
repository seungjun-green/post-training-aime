"""Build the self-contained Colab notebook for DeepSeek s1K regeneration."""

from build_notebooks import ROOT, bootstrap, code, markdown, payload, write_notebook


def cells():
    bundle = payload([ROOT / p for p in [
        "common/__init__.py", "common/io.py", "pipeline/__init__.py",
        "pipeline/regenerate_s1.py", "configs/regenerate_s1_deepseek.yaml",
    ]])
    boot = bootstrap(bundle)
    boot.source = boot.source.replace("/content/lg-korean-aime", "/content/lg-s1-regeneration")
    return [
        markdown(r"""
        # Regenerate s1K reasoning and answers with DeepSeek Pro

        Run on **Colab CPU**. Add `HF_TOKEN` (read access) and `DEEPSEEK_API_KEY` to Colab Secrets
        and enable notebook access. This notebook contains its helper code: no Git push or GPU
        is required. Use a fresh runtime when opening a new version of this notebook.

        Source: `Seungjun/dp_removed_s1K-1.1`, pinned to the same **996 rows** used by the project
        before the SFT length filter. Every original column and row is preserved in the full export.

        DeepSeek Pro solves the original **question only**; it is not given the old reasoning,
        answer, reference solution, or grade. Thinking is enabled. Its native `reasoning_content`
        becomes `deepseek-v4-pro_reasoning`; its final `content` becomes `deepseek-v4-pro_answer`.
        The final response must contain four explicit sections in order: **Planning**,
        **Solution and Evaluation**, **Reflection**, and **Exploration**, followed by
        **Final answer:**. Planning explains the strategy choice. Solution and Evaluation
        contains the full derivation with concrete intermediate checks. Reflection develops a
        concrete refinement and explains what it improves, or explains the specific tradeoff
        for retaining the original approach. It does not require an error. Exploration develops a
        concrete alternative and compares it with the main method even when that method succeeded;
        a full second solution is not required. Instructive errors or dead ends from API thinking
        may be included with their detection and correction, but must not be invented.
        No word or token length target is requested.
        Here "answer" means the API's final `content` field, not a literal `<answer>` wrapper.
        There is **no answer-correctness filter**.

        **The intended new training target is `deepseek-v4-pro_answer`: the complete structured
        worked solution.** `deepseek-v4-pro_reasoning` retains raw API thinking for inspection.
        This notebook does not change the existing SFT loader to use either new column.

        **Updated format:** reopen this notebook in a fresh Colab runtime and rerun the smoke test.
        This replaces the natural-reasoning format with required sections and concrete content
        requirements. It is our adaptation of Kimi's cognitive processes, not a published Kimi
        prompt or a claim that the paper prescribes four fixed sequential stages.
        The new prompt/code creates a fresh output folder, leaving previous results intact.
        Seed 42 selects the same 20 questions, so you can compare the old and updated smoke runs.

        1. Run setup, settings, and source-loading cells.
        2. Run **Smoke test** for 20 reproducibly random examples and download the five-column CSV.
        3. Use the separate **Full run** cell to generate the entire dataset, reusing completed rows.

        A default **Run all** runs the paid smoke test only; `RUN_FULL_GENERATION` starts false.
        Both modes save each completed response immediately to Google Drive. Rerun a cell after
        interruption to resume; unsuccessful rows are attempted again, completed rows are reused.
        Use only **one notebook/runtime at a time** for the same output folder.
        """),
        code('PROJECT_ROOT = "/content/drive/MyDrive/LG-AIME-S1-DeepSeek"'),
        code("""
        # @title Install CPU dependencies, mount Drive, and read secrets
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
            "datasets==5.0.1", "huggingface-hub==0.36.2", "httpx==0.28.1",
            "PyYAML==6.0.3", "tqdm==4.67.1"])
        from google.colab import drive, userdata, files
        drive.mount("/content/drive")
        HF_TOKEN = userdata.get("HF_TOKEN")
        DEEPSEEK_API_KEY = userdata.get("DEEPSEEK_API_KEY")
        if not HF_TOKEN or not DEEPSEEK_API_KEY:
            raise ValueError("Enable notebook access to HF_TOKEN and DEEPSEEK_API_KEY in Secrets")
        """),
        boot,
        markdown("""
        ## Settings and generation prompt

        Defaults come from the bundled `configs/regenerate_s1_deepseek.yaml`, shown below.
        Edit `CONFIG` in this cell before loading the source if needed. The default is
        **`deepseek-v4-pro`, high thinking effort, 16 concurrent requests** with request pacing
        and bounded retries for rate limits and transient API failures. Temperature is omitted
        because DeepSeek ignores it in thinking mode. The sampling seed fixes the 20 selected
        examples; hosted generation itself is not guaranteed deterministic.

        `max_tokens=131072` is an operational ceiling for **reasoning + answer**, not a requested
        length or an SFT limit. It is not included in the prompt. A response that hits this ceiling
        is flagged as incomplete; its raw text is saved in the journal, but is not placed in the
        completed-output columns. No source rows are removed by the old 20,480-token SFT limit.

        Changing the config or bundled generation code creates a separate run folder. Reuse the
        same notebook/settings to resume. Hosted model aliases may change server-side: returned
        model IDs, timestamps, fingerprints, usage, and response IDs are recorded per request.
        """),
        code("""
        import yaml
        CONFIG = yaml.safe_load((CODE_ROOT / "configs/regenerate_s1_deepseek.yaml").read_text())
        # Optional edits go here, before creating the run in the next cell.
        # CONFIG["api"]["concurrency"] = ...
        print(yaml.safe_dump(CONFIG, sort_keys=False, allow_unicode=True))
        """),
        code("""
        # @title Load the pinned Hugging Face source and prepare a resumable run
        import importlib.metadata
        from pipeline.regenerate_s1 import RegenerationRun, load_source, timestamp
        from common.io import append_jsonl, read_jsonl
        source_rows = load_source(CONFIG, HF_TOKEN)
        run = RegenerationRun(source_rows, CONFIG, PROJECT_ROOT)
        runtime_path = run.root / "runtime_sessions.jsonl"
        read_jsonl(runtime_path, repair_tail=True)
        append_jsonl(runtime_path, {"started_at": timestamp(), "python": sys.version,
            "packages": {name: importlib.metadata.version(name) for name in
                         ["datasets", "huggingface-hub", "httpx", "PyYAML", "tqdm"]}})
        print("Source rows:", len(source_rows))
        print("New columns:", run.reasoning_column, run.answer_column)
        print("Smoke sample indices (seeded):", run.smoke_indices)
        print("Drive output folder:", run.root)
        """),
        markdown("""
        ## Smoke test — 20 random examples and a comparison CSV

        This cell makes paid API requests for the 20 sampled rows that are not already complete.
        The CSV has exactly these five columns, in this order:

        `question`, `deepseek_thinking_trajectory`, `deepseek_attempt`,
        `deepseek-v4-pro_reasoning`, `deepseek-v4-pro_answer`.

        UTF-8 CSV preserves multiline text, quotation marks, and Unicode. Long reasoning can exceed
        spreadsheet apps' cell-display limits; the CSV itself retains the full text. Blank generated
        cells mean generation is incomplete/failed/pending or the final-response format is invalid;
        inspect `smoke/status.jsonl` for details. Format checks require each of the four named
        sections exactly once, in order, with nonempty content, followed by a single
        `Final answer:` line and a nonempty conclusion. Markdown or plain headings are accepted.
        Proofs can end with a textual conclusion; no box or length minimum is enforced.
        These checks do not assess correctness or semantic quality. Review the answer column
        using the checklist below; headings and words such as "check" are not quality evidence:

        - Planning: is the method choice justified using this problem's facts or constraints?
        - Solution and Evaluation: is the derivation complete, and do checks show their target,
          calculation or argument, and what the result establishes?
        - Reflection: is a specific limitation identified and a concrete refinement shown,
          with an explanation of what improves? If the original approach is retained, is the
          tradeoff explained specifically rather than merely repeating that the solution is valid?
        - Exploration: does an alternative contain a concrete equation, construction, or
          observation and a comparison with the main method? An unfinished alternative must
          not be presented as independent verification.

        Assess reflection coverage separately from preservation of actual error corrections.
        Compare claimed mistakes and corrections with the raw API reasoning column; a textual
        match alone does not prove internal history. A smooth solution without corrections is
        valid but still needs Reflection and Exploration. Raw returned text,
        including format failures, remains in `attempts.jsonl`. Rerun to retry unsuccessful rows.
        The old `deepseek_grade` is not a grade for the new answer, and no grading is run here.
        """),
        code("""
        # @title Smoke test: generate the fixed random sample and save/download the CSV
        try:
            await run.generate(run.smoke_indices, DEEPSEEK_API_KEY)
        finally:
            smoke_result = run.export(run.smoke_indices, "smoke")
            print(smoke_result["summary"])
            print("CSV saved:", smoke_result["path"])
            print("Row status:", smoke_result["status_path"])
        files.download(str(smoke_result["path"]))
        """),
        markdown("""
        ## Full run — all 996 rows, preserving original columns

        Set `RUN_FULL_GENERATION = True` below to run. There is no baseline-evaluation or smoke-completion gate.
        Matching completed smoke results are reused automatically. After a runtime restart, run
        setup/settings/source-loading, then this cell directly to resume.

        The export is a Hugging Face-compatible **JSONL dataset** in original source order, with
        every original field plus the two generated fields. Existing grades and responses retain
        their original meaning; they are never overwritten. The source HF repository is unchanged.

        `full/dataset.jsonl` is written when every row is complete. Otherwise the export is called
        `full/dataset.partial.jsonl`, still containing every row, with null generated fields for
        unsuccessful rows (including missing, duplicate, empty, or misordered required sections
        and invalid final-answer markers). `full/status.jsonl`
        lists failures by source index; `attempts.jsonl`
        retains raw returned text, including incomplete responses and retry usage. Rerun the cell
        to retry remaining rows. Authentication/balance/invalid-request errors stop the run and
        save partial results. Failed responses are not silently promoted into the training data.

        This notebook only regenerates data. It does not run SFT, evaluation, or HF publication.
        """),
        code("""
        # @title Full run: generate all rows (completed smoke rows are reused)
        RUN_FULL_GENERATION = False
        if RUN_FULL_GENERATION:
            full_indices = list(range(len(source_rows)))
            try:
                await run.generate(full_indices, DEEPSEEK_API_KEY)
            finally:
                full_result = run.export(full_indices, "full")
                print(full_result["summary"])
                print("Dataset saved:", full_result["path"])
                print("Row status:", full_result["status_path"])
            if full_result["summary"]["remaining"]:
                print("Some rows remain incomplete. Inspect status.jsonl and rerun this cell.")
            else:
                print("All source rows now have generated reasoning and answers.")
        else:
            print("Full run is off. Enable RUN_FULL_GENERATION to process all source rows.")
        """),
        markdown("""
        Load the finished export later with
        `load_dataset("json", data_files=".../full/dataset.jsonl", split="train")`.
        Drive stores the dataset and journals; save a notebook copy in Drive to retain cell output
        displays too. The API journal records returned token usage, including retries. A request
        whose response was lost or cancelled can still be billed; consult the provider dashboard.

        References: [Kimi k1.5, §2.2](https://arxiv.org/html/2501.12599v1#S2.SS2),
        [DeepSeek thinking fields](https://api-docs.deepseek.com/guides/thinking_mode/),
        [DeepSeek API](https://api-docs.deepseek.com/api/create-chat-completion/).
        """),
    ]


if __name__ == "__main__":
    write_notebook("regenerate_s1_deepseek.ipynb", cells())
