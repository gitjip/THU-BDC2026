import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import docker
import pandas as pd


DATA_ROOT_PATH = Path("./")
TEST_DIR = DATA_ROOT_PATH / "test"
TEMP_DIR = DATA_ROOT_PATH / "temp"
OUTPUT_DIR = TEST_DIR / "output"
RESULTS_OUTPUT_DIR = TEST_DIR / "results_output"
TEMP_SCORE_PATH = TEMP_DIR / "latest_score.csv"


def delete_file() -> None:
	output_target = TEST_DIR / "output"
	temp_target = TEMP_DIR

	if output_target.exists():
		for item in output_target.iterdir():
			if item.is_file() or item.is_symlink():
				item.unlink(missing_ok=True)
			elif item.is_dir():
				shutil.rmtree(item, ignore_errors=True)

	if temp_target.exists():
		for item in temp_target.iterdir():
			if item.is_file() or item.is_symlink():
				item.unlink(missing_ok=True)
			elif item.is_dir():
				shutil.rmtree(item, ignore_errors=True)


def run_docker_rm() -> None:
	command = ["docker", "rm", "test-app-1"]
	subprocess.run(command, check=True)
	print("已删除 test-app-1 容器。")


def run_docker_rmi() -> None:
	command = ["docker", "rmi", "bdc2026"]
	subprocess.run(command, check=True)
	print("已删除 bdc2026 镜像。")


def run_docker_load(tar_file: Path) -> None:
	command = ["docker", "load", "-i", str(tar_file)]
	print(" ".join(command))
	subprocess.run(command, check=True)
	print(f"已从 {tar_file} 加载镜像。")


def run_docker_compose_up(tar_name: str, timeout_seconds: int = 60 * 60 * 8 + 300) -> None:
	up_command = ["docker", "compose", "-p", "test", "up", "-d"]
	subprocess.run(up_command, check=True)
	print("已启动 docker compose。")

	elapsed = 0
	client = docker.from_env()
	container = client.containers.get("test-app-1")

	while True:
		time.sleep(30)
		elapsed += 30
		if elapsed > timeout_seconds:
			print("超过8小时5分钟，停止运行")
			break
		container.reload()
		if container.status == "exited":
			print("容器已退出，停止运行")
			break

	down_command = ["docker", "compose", "down"]
	subprocess.run(down_command, check=True)
	print("已停止 docker compose。")

	RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	src = OUTPUT_DIR / "result.csv"
	dst = RESULTS_OUTPUT_DIR / f"{tar_name}.csv"
	shutil.copy2(src, dst)
	print(f"已复制预测结果到 {dst}。")


def run_score(file_name: str) -> None:
	team_name = Path(file_name).stem
	command = [sys.executable, "test/score_docker.py", team_name]
	subprocess.run(command, check=True)
	print("已完成 score_docker.py 评分。")

	df1 = pd.read_csv(TEMP_SCORE_PATH)
	print(df1)
	try:
		df2 = pd.read_csv("./test/result.csv")
	except FileNotFoundError:
		df2 = pd.DataFrame(columns=["Team Name", "Final Score"])

	df_combined = pd.concat([df2, df1], ignore_index=True)
	df_combined.to_csv("./test/result.csv", mode="w", index=False)

	print("已追加评分结果到 ./test/result.csv。")


def read_tar_files(input_file: Path) -> list[str]:
	with open(input_file, "r", encoding="utf-8") as file:
		return [line.strip() for line in file.readlines() if line.strip()]


def main() -> None:
	input_file = TEST_DIR / "tar_files_list.txt"
	tar_files = read_tar_files(input_file)

	print("tar_files已成功从文件中读取：")
	print(tar_files)

	for tar_file_name in tar_files:
		print(tar_file_name)
		tar_name = Path(tar_file_name).stem
		tar_file_path = TEST_DIR / "tars" / tar_file_name

		try:
			delete_file()
		except Exception as error:
			print(f"delete_file Error occurred: {error}")

		try:
			run_docker_rm()
		except Exception as error:
			print(f"run_docker_rm Error occurred: {error}")

		try:
			run_docker_rmi()
		except Exception as error:
			print(f"run_docker_rmi Error occurred: {error}")

		try:
			run_docker_load(tar_file_path)
		except Exception as error:
			print(f"run_docker_load Error occurred: {error}")

		try:
			run_docker_compose_up(tar_name)
		except Exception as error:
			print(f"run_docker_compose_up Error occurred: {error}")

		try:
			run_score(tar_file_name)
		except Exception as error:
			print(f"run_score Error occurred: {error}")


if __name__ == "__main__":
	main()
