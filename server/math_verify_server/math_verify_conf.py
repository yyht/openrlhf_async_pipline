import os
import multiprocessing

RANK = os.getenv('RANK', '0')
LOG_DIR = os.getenv('LOG_DIR', 'xxxxxxx')

daemon=True # 设置守护进程
chdir='/cpfs/user/debug' # 工作目录
worker_class='uvicorn.workers.UvicornWorker' # 工作模式
workers=multiprocessing.cpu_count()+1  # 并行工作进程数 核心数*2+1个
worker_connections = 500  # 设置最大并发量
loglevel='debug' # 错误日志的日志级别
preload_app = True
access_log_format = '%(t)s %(p)s %(h)s "%(r)s" %(s)s %(L)s %(b)s %(f)s" "%(a)s"'
# 设置访问日志和错误信息日志路径

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_dir = f"{LOG_DIR}/code_{RANK}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

pidfile=f'{log_dir}/gunicorn.pid'  # 设置进程文件目录
accesslog = f"{log_dir}/gunicorn_access.log"
errorlog = f"{log_dir}/gunicorn_error.log"