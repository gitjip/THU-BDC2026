import pandas as pd
import subprocess
import docker
import time
import os
import sys
DATA_ROOT_PATH = "./"
TEMP_SCORE_PATH = "./temp/latest_score.csv"


def get_docker_client():
    docker_host = os.environ.get("DOCKER_HOST")
    if docker_host:
        return docker.from_env()

    default_sock = "/var/run/docker.sock"
    if os.path.exists(default_sock):
        return docker.from_env()

    desktop_sock = os.path.expanduser("~/.docker/run/docker.sock")
    if os.path.exists(desktop_sock):
        os.environ["DOCKER_HOST"] = f"unix://{desktop_sock}"
        return docker.from_env()

    raise RuntimeError(
        "Docker socket not found. Please start Docker Desktop or set DOCKER_HOST manually."
    )


def delete_file():
    command = f"rm -rf {DATA_ROOT_PATH}/test/output/*"
    subprocess.run(command, shell=True, check=True)
    command = f"rm -rf {DATA_ROOT_PATH}/temp/*"
    subprocess.run(command, shell=True, check=True)


# 运行docker rm命令
def run_docker_rm():
    # command = "docker stop $(docker ps -aq)"
    # subprocess.run(command, shell=True, check=True)
    command = "docker rm test-app-1"
    # command = "docker rm $(docker ps -aq) -f"
    subprocess.run(command, shell=True, check=True)
    print("已删除 test-app-1 容器。")


# 运行docker rmi命令
def run_docker_rmi():
    command = "docker rmi bdc2026"
    # command = "docker rmi $(docker images -aq) -f"
    subprocess.run(command, shell=True, check=True)
    print("已删除 bdc2026 镜像。")


# 运行docker load命令
def run_docker_load(tar_file):
    command = f"docker load -i {tar_file}"
    print(command)
    subprocess.run(command, shell=True, check=True)
    print(f"已从 {tar_file} 加载镜像。")


# 运行docker-compose up命令
def run_docker_compose_up(tar_name):
    # command = "docker compose up"
    command = "docker compose -p test up -d"
    subprocess.run(command, shell=True, check=True)
    print("已启动 docker compose。")

    sumtm = 0
    client = get_docker_client()
    container = client.containers.get("test-app-1")
    while True:
        time.sleep(30)
        sumtm += 30
        if sumtm > 60 * 60 * 8 + 300:  # 超过8小时+5分钟
            print("超过8小时5分钟，停止运行")
            break
        container.reload()
        if container.status == "exited":
            print("容器已退出，停止运行")
            break
    command = "docker compose down"
    subprocess.run(command, shell=True, check=True)
    print("已停止 docker compose。")

    command = f"cp {DATA_ROOT_PATH}/test/output/result.csv ./test/results_output/{tar_name}.csv"
    subprocess.run(command, shell=True, check=True)
    print(f"已复制预测结果到 ./test/results_output/{tar_name}.csv。")


# 运行score.sh命令
def run_score(file_name):
    team_name = file_name.split("/")[-1].split(".")[0]  # 获取文件名作为team_name
    command = [sys.executable, "test/score_docker.py", team_name]
    subprocess.run(command, check=True)
    print("已完成 score_docker.py 评分。")

    df1 = pd.read_csv(TEMP_SCORE_PATH)
    print(df1)
    try:
        df2 = pd.read_csv("./test/result.csv")
    except FileNotFoundError:
        print("未找到 ./test/result.csv，将创建新的本地汇总评分表。")
        df2 = pd.DataFrame(columns=["Team Name", "Final Score"])

    # print(df1)
    # print(df2)
    df_combined = pd.concat([df2, df1], ignore_index=True)

    df_combined.to_csv("./test/result.csv", mode="w", index=False)

    print("已追加评分结果到 ./test/result.csv。")


# 输出文件路径
input_file = "./test/tar_files_list.txt"

# 从文件中读取tar_files
tar_files = []
with open(input_file, "r", encoding="utf-8") as file:
    tar_files = [line.strip() for line in file.readlines()]  # 去掉每行的换行符

print("tar_files已成功从文件中读取：")
print(tar_files)

# 打印所有找到的.tar文件
for tar_file in tar_files:
    print(tar_file)
    tar_name = tar_file.split(".")[0]  # 获取文件名
    tar_file = "./test/tars/" + tar_file  # 确保路径正确
    try:
        delete_file()
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
    try:
        run_docker_rm()
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
    try:
        run_docker_rmi()
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
    try:
        run_docker_load(tar_file)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
    try:
        run_docker_compose_up(tar_name)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
    try:
        run_score(tar_file)  # 需要传递tar_file作为参数
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
    # command = f"del {tar_file}"
    # subprocess.run(command, shell=True, check=True)
