import subprocess
import sys


def _run_module(module: str) -> None:
    print(f"[refresh_job] Running: {sys.executable} -m {module}", flush=True)
    subprocess.check_call([sys.executable, "-m", module])


def main() -> int:
    try:
        _run_module("transform.build_silver_games")
        _run_module("features.build_features")
    except subprocess.CalledProcessError as exc:
        print(
            f"[refresh_job] Failed with exit code {exc.returncode}: {exc.cmd}",
            flush=True,
        )
        return exc.returncode
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[refresh_job] Unexpected error: {exc}", flush=True)
        return 1

    print("[refresh_job] Completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
