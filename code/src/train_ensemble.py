import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from data_utils import setup_logging
from ensemble_config import get_submission_ensemble_sources


REPO_ROOT = Path(__file__).resolve().parents[2]
logger = setup_logging("bdc.train_ensemble", REPO_ROOT / "model" / "ensemble_train.log")


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练正式提交用的两模型集成源模型")
    parser.add_argument("--debug", action="store_true", help="使用小模型快速检查集成训练流程")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要训练的源模型配置")
    return parser.parse_args()


def merged_env(source_env: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(source_env)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_source_train(label: str, env: dict[str, str]) -> float:
    output_dir = Path(env["BDC_OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "code/src/train.py"]
    logger.info("")
    logger.info("=" * 72)
    logger.info("训练集成源模型: %s -> %s", label, output_dir)
    logger.info("=" * 72)
    start = time.perf_counter()
    subprocess.run(command, cwd=REPO_ROOT, env=merged_env(env), check=True)
    duration = time.perf_counter() - start
    logger.info("源模型完成: %s | 耗时=%s", label, format_duration(duration))
    return duration


def main() -> None:
    args = parse_args()
    debug = args.debug or os.environ.get("BDC_FAST_DEV", "").strip().lower() in {"1", "true", "yes", "on"}
    sources = get_submission_ensemble_sources(debug=debug)

    logger.info("提交集成训练模式: %s", "debug" if debug else "official")
    logger.info("源模型数量: %s", len(sources))
    for source in sources:
        logger.info("源模型: %s | output_dir=%s", source.label, source.output_dir)
        for key in sorted(source.env):
            logger.info("  %s=%s", key, source.env[key])

    if args.dry_run:
        return

    start = time.perf_counter()
    total_source_seconds = 0.0
    for source in sources:
        total_source_seconds += run_source_train(source.label, source.env)
    total_seconds = time.perf_counter() - start
    logger.info("")
    logger.info("提交集成训练完成: 源模型耗时合计=%s | 总耗时=%s", format_duration(total_source_seconds), format_duration(total_seconds))


if __name__ == "__main__":
    main()
