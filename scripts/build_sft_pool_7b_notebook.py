"""Build the self-contained Colab for the 7B follow-up on the 3B 0/1/2 subset."""
from build_notebooks import ROOT, code, markdown, payload, write_notebook
from build_sft_pool_sampling_notebook import FILES


def cells():
    files = FILES + ['pipeline/sample_sft_pool_7b.py', 'configs/sft_pool_sampling_7b.yaml']
    return [
        markdown('''
        # 3B에서 0·1·2개 맞힌 문제 → Math-7B-Instruct 8회 생성

        **Colab GPU 런타임(A100 권장)을 선택하고, Secrets에 쓰기 권한이 있는
        `HF_TOKEN`을 등록한 다음 모두 실행하세요.** 별도 smoke 없이 전체 대상에 실행합니다.

        입력 및 업로드: **Seungjun/clean-math-sft-pool-30k**.
        기존 **`num_correct`가 0, 1, 2인 행만** Qwen/Qwen2.5-Math-7B-Instruct로
        8번 생성하고 기존 Math-Verify 채점 코드(`other` 표기 처리 포함)로 채점합니다.

        새 열: `7B_responses`, `7B_extracted_answers`, `7B_correct`,
        `7B_response_tokens`, `7B_finish_reasons`, `7B_num_correct`.
        앞의 다섯 열은 각각 길이 8의 리스트, 마지막 열은 0–8의 정수입니다.
        **전체 원본 행과 기존 3B 결과는 유지**하며, 대상이 아닌 행의 새 열은
        `null`(미실행)입니다. 추가적인 문제 제외나 정답 수정은 하지 않습니다.

        처음 실행 시 HF 데이터 버전을 고정합니다. 각 문제 완료 시 Drive에 저장하며,
        중단되면 동일한 `RUN_NAME`, 설정, GPU 종류로 다시 실행하여 이어갑니다.
        완료 후 검증을 거쳐 같은 HF 데이터셋의 기본 train 데이터를 자동 갱신합니다.
        '''),
        code('''
        RUN_NAME = 'qwen25-math-7b-instruct-eight-3b-012-v1'
        DRIVE_ROOT = '/content/drive/MyDrive/LG-SFT-Pool-7B'
        CODE_ROOT = '/content/lg-sft-pool-7b'
        EVAL_ENV = '/content/lg-eval-env'
        TEMPERATURE = 1.0
        TOP_P = 1.0
        MAX_NEW_TOKENS = 3072
        MAX_NUM_SEQS = 8
        SEED = 42
        UPLOAD_TO_HF = True
        '''),
        markdown('## 1. Drive 연결 · 코드 및 실행 환경 설치'),
        code(f'''
        import base64, io, json, os, subprocess, sys, zipfile
        from pathlib import Path
        from google.colab import drive, userdata
        drive.mount('/content/drive')
        os.environ['HF_TOKEN'] = userdata.get("HF_TOKEN")
        if not os.environ['HF_TOKEN']:
            raise ValueError('Enable a write-capable HF_TOKEN in Colab Secrets')
        Path(CODE_ROOT).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode({payload([ROOT / p for p in files])!r}))) as bundle:
            bundle.extractall(CODE_ROOT)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.11.22', 'PyYAML==6.0.3'])
        subprocess.check_call([sys.executable, 'scripts/setup_eval_runtime.py', '--venv', EVAL_ENV], cwd=CODE_ROOT)
        sys.path.insert(0, CODE_ROOT)
        from common.process import run_logged
        '''),
        markdown('''
        ## 2. 채점 검사 · HF 버전 고정 · 대상 행 확인

        GPU 생성 전에 기존 채점 검사를 실행합니다. 아래 로그에 원본 행 수,
        대상 행 수와 총 생성 답변 수가 출력됩니다. 모델과 데이터의 버전 및 설정은
        `manifest.json`에 저장됩니다. 설정을 바꾸려면 새 `RUN_NAME`을 사용하세요.
        '''),
        code('''
        import yaml
        run_logged([str(Path(EVAL_ENV) / 'bin/python'), '-m', 'eval.pool_grading_checks'],
                   cwd=CODE_ROOT, log_path=Path(CODE_ROOT) / 'grading_checks.log')
        CONFIG = yaml.safe_load((Path(CODE_ROOT) / 'configs/sft_pool_sampling_7b.yaml').read_text())
        CONFIG['sampling'].update(temperature=TEMPERATURE, top_p=TOP_P,
                                  max_new_tokens=MAX_NEW_TOKENS, seed=SEED)
        CONFIG['engine']['max_num_seqs'] = MAX_NUM_SEQS
        RUN_DIR = Path(DRIVE_ROOT) / RUN_NAME
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH = RUN_DIR / 'requested_config.yaml'
        CONFIG_PATH.write_text(yaml.safe_dump(CONFIG, sort_keys=False))
        COMMAND = [str(Path(EVAL_ENV) / 'bin/python'), '-m', 'pipeline.sample_sft_pool_7b',
                   '--config', str(CONFIG_PATH), '--run-dir', str(RUN_DIR)]
        run_logged(COMMAND + ['--mode', 'prepare'], cwd=CODE_ROOT,
                   log_path=RUN_DIR / 'prepare_console.log')
        manifest = json.loads((RUN_DIR / 'manifest.json').read_text())
        print('원본 행:', manifest['source_rows'])
        print('7B 대상 행:', len(manifest['selected_indices']))
        print('7B 생성 답변 수:', len(manifest['selected_indices']) * 8)
        print('저장 위치:', RUN_DIR)
        '''),
        markdown(r'''
        ## 3. 대상 행마다 8개 생성 · 채점 · 전체 데이터에 새 열 추가

        모델의 공식 CoT 시스템 지시문과 native chat template을 사용합니다.
        답변은 `\boxed{}`에 쓰도록 지시하고, `<|im_end|>`(151645)를 EOS로
        사용하며 `<|endoftext|>`(151643)에서도 멈춥니다. EOS 무시는 꺼져 있습니다.

        이 모델의 기본 문맥 길이는 **입력+출력 합계 4,096토큰**입니다.
        답변은 기본 최대 3,072토큰이며, 긴 입력에서는 남은 문맥 길이에 맞춰
        출력 한도를 줄입니다. 질문을 잘라내지 않습니다. 한도까지 생성한 답은
        `7B_finish_reasons="length"`로 기록됩니다. 정확한 입력 길이와 출력 한도는
        `grading_audit.jsonl`에 저장됩니다.

        생성 답변은 `responses_7b.jsonl`에 문제별로 저장됩니다. 완료 후 기존 열과
        새 열이 합쳐진 전체 데이터는 `full/data/*.parquet`에 저장됩니다.
        '''),
        code('''
        run_logged(COMMAND + ['--mode', 'full'], cwd=CODE_ROOT,
                   log_path=RUN_DIR / 'full_console.log')
        summary = json.loads((RUN_DIR / 'full/summary.json').read_text())
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print('전체 결과:', RUN_DIR / 'full/data')
        print('7B 채점 상세:', RUN_DIR / 'full/grading_audit.jsonl')
        '''),
        markdown('''
        ## 4. HF 데이터셋 갱신

        모든 대상 행의 8개 결과가 완성되고 기존 행/3B 결과가 그대로인지 검증한 후
        **Seungjun/clean-math-sft-pool-30k**에 업로드합니다. 원본 파일을 보존하고
        기본 train 경로를 새 Parquet 파일로 변경합니다. 실행 도중 다른 작업이 HF를
        갱신한 경우 덮어쓰지 않고 중단합니다. Drive 결과는 계속 남습니다.
        '''),
        code('''
        if UPLOAD_TO_HF:
            run_logged(COMMAND + ['--mode', 'upload'], cwd=CODE_ROOT,
                       log_path=RUN_DIR / 'upload_console.log')
            receipt = json.loads((RUN_DIR / 'upload_receipt.json').read_text())
            print('업로드 및 검증 완료:', receipt['url'])
        else:
            print('Drive에 저장 완료:', RUN_DIR / 'full/data')
        '''),
        markdown('''
        Model instructions and pinned configuration:
        [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct),
        [model config](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct/blob/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d/config.json).
        Grader: [Hugging Face Math-Verify](https://github.com/huggingface/Math-Verify).
        '''),
    ]


if __name__ == '__main__':
    write_notebook('sample_sft_pool_math7b_8.ipynb', cells())
