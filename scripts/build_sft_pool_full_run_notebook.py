"""Build the self-contained full-run Colab with the current sampling/grading bundle."""
from build_notebooks import ROOT, code, markdown, payload, write_notebook
from build_sft_pool_sampling_notebook import FILES
from build_sft_pool_sampling_notebook import cells as sampling_cells


def cells(include_resume=True):
    # Share the exact dependency installation and embedded code with the smoke notebook.
    install = sampling_cells()[3]
    install.source = install.source.replace(
        payload([ROOT / p for p in FILES]),
        payload([ROOT / p for p in FILES + ['common/pool_progress.py'] + (['common/pool_resume.py'] if include_resume else [])]))
    install.source += '\nimport importlib\nimport common.pool_progress as pool_progress\nimportlib.reload(pool_progress)\nrun_logged = pool_progress.run_pool_logged'
    result = [
        markdown('''
        # Qwen2.5-3B 전체 실행 — 문제당 8개 생성 · 채점 · HF 업로드

        **Colab에서 GPU 런타임을 선택하고, Secrets에 쓰기 권한이 있는 `HF_TOKEN`을
        등록한 다음 모두 실행하세요.** L4/A100 이상의 GPU를 권장합니다.
        별도 smoke 생성 없이 전체 실행 후 자동으로 HF에 업로드합니다.

        입력 및 업로드 대상: **Seungjun/clean-math-sft-pool-30k**.
        처음 실행할 때 최신 텍스트 정리 완료 버전을 고정합니다.
        기존 품질 필터를 통과한 모든 문제에 Qwen/Qwen2.5-3B **base**로
        답을 8개씩 생성하고 Math-Verify 0.9.0으로 채점합니다.
        `answer_type="other"`의 시간·비율·진법·퍼센트 등 표기 처리 수정도 포함됩니다.

        `synthetic_math` 출처만으로 문제를 제외하지 않습니다. 다른 품질 규칙은 유지됩니다.
        이전 필터로 실행한 체크포인트와 대상 행이 달라지므로 새 RUN_NAME을 사용하세요.

        결과에는 `responses`, `extracted_answers`, `correct`, `response_tokens`,
        `finish_reasons`, `num_correct`가 추가됩니다.
        **0/8인 행도 이번 전체 결과에는 저장됩니다.** 나중에 `num_correct > 0`으로
        필요한 행을 선택할 수 있습니다.

        문제 하나를 끝낼 때마다 Drive에 저장합니다. 연결이 끊기면 같은 설정과
        `RUN_NAME`으로 다시 실행하여 이어서 진행하세요. 설정/코드/GPU 종류가 바뀌면
        이어 실행이 거부될 수 있습니다. 새 실험에는 새 `RUN_NAME`을 사용하세요.
        '''),
        code('''
        RUN_NAME = 'qwen25-3b-fresh-run-002'
        DRIVE_ROOT = '/content/drive/MyDrive/LG-SFT-Pool-Sampling'
        CODE_ROOT = '/content/lg-sft-pool-sampling'
        EVAL_ENV = '/content/lg-eval-env'
        TEMPERATURE = 1.0
        TOP_P = 1.0
        MAX_NEW_TOKENS = 8192
        MAX_NUM_SEQS = 16  # GPU 메모리가 부족하면 새 실행에서 줄이세요.
        SEED = 42
        UPLOAD_FULL = True  # 완료 후 같은 HF 데이터셋에 자동 업로드
        '''),
        markdown('## 1. Drive 연결 및 실행 환경 설치'),
        install,
        markdown('''
        ## 2. 채점 확인 및 입력 데이터 준비

        짧은 CPU 채점 검사를 실행하고 HF 데이터 버전을 고정합니다.
        기존 품질 필터와 검토된 정답 수정 규칙을 그대로 적용합니다.
        제외/수정 내역은 `quality_decisions.jsonl`, 개수는 `quality_summary.json`에
        저장됩니다. 준비 코드가 smoke용 ID 목록도 기록하지만 답변 생성은 전체 실행에서만 합니다.
        '''),
        code('''
        pool_progress.run_pool_logged([str(Path(EVAL_ENV) / 'bin/python'), '-m', 'eval.pool_grading_checks'],
                   cwd=CODE_ROOT, log_path=Path(CODE_ROOT) / 'grading_checks.log')
        import yaml
        CONFIG = yaml.safe_load((Path(CODE_ROOT) / 'configs/sft_pool_sampling.yaml').read_text())
        CONFIG['data'].update(revision='main', expected_rows=None, require_text_cleanup=True)
        CONFIG['sampling'].update(temperature=TEMPERATURE, top_p=TOP_P,
                                  max_new_tokens=MAX_NEW_TOKENS, seed=SEED)
        CONFIG['engine']['max_num_seqs'] = MAX_NUM_SEQS
        CONFIG['smoke']['seed'] = SEED
        RUN_DIR = Path(DRIVE_ROOT) / RUN_NAME
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH = RUN_DIR / 'requested_config.yaml'
        CONFIG_PATH.write_text(yaml.safe_dump(CONFIG, sort_keys=False))
        COMMAND = [str(Path(EVAL_ENV) / 'bin/python'), '-m', 'pipeline.sample_sft_pool',
                   '--config', str(CONFIG_PATH), '--run-dir', str(RUN_DIR)]
        subprocess.check_call([str(Path(EVAL_ENV) / 'bin/python'), '-m', 'common.pool_progress',
                               '--check-resume', str(CONFIG_PATH), str(RUN_DIR)], cwd=CODE_ROOT)
        pool_progress.run_pool_logged(COMMAND + ['--mode', 'prepare'], cwd=CODE_ROOT,
                   log_path=RUN_DIR / 'prepare_console.log')
        manifest = json.loads((RUN_DIR / 'manifest.json').read_text())
        print('고정된 HF 버전:', manifest['dataset_revision'])
        print('HF 입력 행 수:', manifest['source_rows'])
        quality = json.loads((RUN_DIR / 'quality_summary.json').read_text())
        print(f"Eligible problems: {quality['eligible_rows']:,} / {quality['input_rows']:,}")
        print('Quality report:', RUN_DIR / 'quality_summary.json')
        print('저장 위치:', RUN_DIR)
        '''),
        markdown('''
        ## 3. 전체 생성 및 채점

        문제당 8개, temperature=1.0, top-p=1.0, 답변당 최대 8,192토큰입니다.
        Qwen의 `<|endoftext|>`(151643), `<|im_end|>`(151645)에서 생성을 멈춥니다.
        토큰 한도에 도달하면 `finish_reasons`에 `length`가 기록됩니다.
        전체 생성에는 오랜 시간이 걸릴 수 있습니다. 진행 상황은 완료 수·경과 시간·예상 남은 시간을 보여주는 tqdm 진행 막대에 표시됩니다. 전체 로그는 full_console.log에 보존됩니다.
        '''),
        code('''
        pool_progress.run_pool_logged(COMMAND + ['--mode', 'full'], cwd=CODE_ROOT,
                   log_path=RUN_DIR / 'full_console.log')
        summary = json.loads((RUN_DIR / 'full/summary.json').read_text())
        print(f"Completed: {summary['rows']:,} problems / {summary['responses']:,} responses")
        print(f"Response accuracy: {summary['accuracy_per_response']:.2%} | At least one correct: {summary['at_least_one_correct_fraction']:.2%}")
        print(f"Truncated: {summary['truncated_fraction']:.2%} | Grading exceptions: {summary['grading_exceptions']:,}")
        print('Full summary:', RUN_DIR / 'full/summary.json')
        print('전체 결과 Parquet:', RUN_DIR / 'full/data')
        print('채점 상세 기록:', RUN_DIR / 'full/grading_audit.jsonl')
        '''),
        markdown('''
        ## 4. 완료된 전체 결과를 HF에 업로드

        모든 대상 행의 생성이 끝나야 업로드할 수 있습니다.
        **Seungjun/clean-math-sft-pool-30k**의 기본 train 데이터를 이번 결과로
        갱신하며 기존 원본 파일은 보존합니다. 생성 도중 HF 데이터가 다른 버전으로
        변경되었다면 덮어쓰지 않고 중단합니다. Drive의 결과는 유지됩니다.
        '''),
        code('''
        if UPLOAD_FULL:
            pool_progress.run_pool_logged(COMMAND + ['--mode', 'upload'], cwd=CODE_ROOT,
                       log_path=RUN_DIR / 'upload_console.log')
            receipt = json.loads((RUN_DIR / 'upload_receipt.json').read_text())
            print('업로드 및 검증 완료:', receipt['url'])
        else:
            print('결과는 Drive에 저장되었습니다:', RUN_DIR / 'full/data')
        '''),
    ]
    if include_resume:
        result[1].source = result[1].source.replace(
            "DRIVE_ROOT =", "RUN_MODE = 'auto'  # auto: resume this RUN_NAME if found; resume: require it; new: never overwrite.\nDRIVE_ROOT =", 1)
        result[0].source += "\n\nThis notebook defaults to resuming qwen25-3b-fresh-run-002 (RUN_MODE='auto'). Keep that run name and Run all to continue its saved progress. Saved settings are restored automatically. RUN_MODE='resume' requires an existing run; 'new' requires an empty directory."
        result[4].source += "\n\nSaved runs restore their settings and pinned input. Code/filter changes still require the original compatible notebook or a new run name. A disconnected session regenerates only unfinished problems; completed problems are reused. Stop other generation sessions before resuming. Unreadable checkpoint records are backed up and removed only after all surviving records pass source/annotation validation; missing problems are regenerated. Do not launch two sessions on the same run directory."
        result[3].source += '\nimport common.pool_resume as pool_resume\nimportlib.reload(pool_resume)'
        result[5].source = result[5].source.replace(
            'RUN_DIR.mkdir(parents=True, exist_ok=True)',
            'CONFIG = pool_resume.resolve_run_config(CONFIG, RUN_DIR, RUN_MODE)\nRUN_DIR.mkdir(parents=True, exist_ok=True)')
        result[5].source += '\nrun_setup_logged([str(Path(EVAL_ENV) / "bin/python"), "-m", "common.pool_resume",\n                  "--recover", str(CONFIG_PATH), str(RUN_DIR)],\n                 cwd=CODE_ROOT, log_path=RUN_DIR / "checkpoint_recovery.log")\npool_resume.show_checkpoint_status(RUN_DIR)'
        result[7].source = result[7].source.replace(
            "COMMAND + ['--mode', 'full']",
            "[str(Path(EVAL_ENV) / 'bin/python'), '-m', 'common.pool_resume', '--generate', str(CONFIG_PATH), str(RUN_DIR)]")
    return result


if __name__ == '__main__':
    write_notebook('full_run_clean_sft_pool_qwen3b_8.ipynb', cells())
