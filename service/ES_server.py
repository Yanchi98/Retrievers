import subprocess
import os
from config import es_dir, es_user, start_command


# 切换到指定目录
os.chdir(es_dir)

# 使用su命令切换用户并启动服务
command = f"su {es_user} -c {start_command}"

try:
    # 启动Elasticsearch服务
    subprocess.run(command, shell=True, check=True)
    print("Elasticsearch服务已启动")
except subprocess.CalledProcessError as e:
    print(f"启动服务时出错: {e}")
