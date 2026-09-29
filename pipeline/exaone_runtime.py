"""Manage only the notebook's own loopback-only vLLM process."""

import asyncio
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import httpx

from common.io import write_json


def server_command(python, config):
    return [
        str(python),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        config["TRANSLATION_MODEL"],
        "--revision",
        config["MODEL_REVISION"],
        "--tokenizer-revision",
        config["MODEL_REVISION"],
        "--served-model-name",
        config["TRANSLATION_MODEL"],
        "--host",
        "127.0.0.1",
        "--port",
        str(config["SERVER_PORT"]),
        "--reasoning-parser",
        "qwen3",
        "--limit-mm-per-prompt",
        '{"image": 0}',
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(config["MAX_MODEL_LEN"]),
        "--gpu-memory-utilization",
        str(config["GPU_MEMORY_UTILIZATION"]),
        "--max-num-seqs",
        str(config["MAX_CONCURRENCY"]),
        "--max-num-batched-tokens",
        "2048",
        "--enable-chunked-prefill",
        "--generation-config",
        "vllm",
        "--seed",
        str(config["SEED"]),
        "--enforce-eager",
        "--disable-log-requests",
    ]


def stop_server(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


async def start_server(python, config, token):
    # Do not attach to or terminate an unknown process already using the port.
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", config["SERVER_PORT"]))
        except OSError as exc:
            raise RuntimeError(
                "Server port is occupied; stop your old server or change SERVER_PORT"
            ) from exc
    root = Path(config["PROJECT_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "server_identity.json").unlink(missing_ok=True)
    log = root / "exaone_server.log"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0",
        HF_TOKEN=token,
        VLLM_NO_USAGE_STATS="1",
        DO_NOT_TRACK="1",
    )
    with log.open("a") as stream:
        process = subprocess.Popen(
            server_command(python, config),
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    deadline = time.monotonic() + config["SERVER_STARTUP_TIMEOUT_SECONDS"]
    next_update = 0
    try:
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"EXAONE server exited ({process.returncode}). See {log}")
                try:
                    response = await client.get(f"http://127.0.0.1:{config['SERVER_PORT']}/health")
                    if response.status_code == 200:
                        print("EXAONE ready. Thinking is disabled on every translation request.")
                        write_json(
                            root / "server_identity.json",
                            {
                                k: config[k]
                                for k in ["TRANSLATION_MODEL", "MODEL_REVISION", "MAX_MODEL_LEN"]
                            },
                        )
                        return process
                except httpx.TransportError:
                    pass
                if time.monotonic() >= next_update:
                    print(f"Loading EXAONE weights / starting GPU engine. Log: {log}", flush=True)
                    next_update = time.monotonic() + 30
                await asyncio.sleep(2)
        raise TimeoutError(f"EXAONE startup timed out; inspect {log}, then rerun the server cell")
    except BaseException:
        stop_server(process)
        raise
