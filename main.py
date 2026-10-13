"""Model server entry point: vLLM's OpenAI-compatible server for one model from configs/serve.toml.

    SRV_NAME=gen python main.py        (started by serve.sh)

The model and its options come from the config, not the command line, and every process title vLLM would
set (APIServer, EngineCore, workers) is replaced by PROC_TITLE, so on a shared machine ps and nvidia-smi
show only `python main.py`.
"""
import os
import sys


def _pin_process_title() -> None:
    title = os.environ.get("PROC_TITLE", "python main.py")
    try:
        import setproctitle
    except ImportError:
        return
    original = setproctitle.setproctitle
    setproctitle.setproctitle = lambda *_args, **_kwargs: original(title)
    original(title)


# Top level on purpose: spawned child processes re-import this module (as __mp_main__) before vLLM
# retitles them, so the patch is in place in the engine core too.
_pin_process_title()


def main() -> None:
    from core import env
    from core.config import load_cfg

    env.set_defaults()
    spec = load_cfg(env.BASE / "configs/serve.toml")["servers"][os.environ["SRV_NAME"]]
    sys.argv = ["vllm", "serve", *spec["args"], "--host", "127.0.0.1", "--port", str(spec["port"])]
    from vllm.entrypoints.cli.main import main as cli
    cli()


if __name__ == "__main__":
    main()
