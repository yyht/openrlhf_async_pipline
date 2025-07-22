

from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import gc, os, time
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

INTERVAL = int(os.getenv('GC_INTERVAL', '3600'))
RANK = int(os.getenv('RANK', '0'))

TOTAL_INTERVAL = INTERVAL + 60 * RANK

logger.info({
    'INFO': "##GC##",
    'VALUE': TOTAL_INTERVAL
})


def gc_task():
    start = time.time()
    gc.collect(2)  
    _, gen1, gen2 = gc.get_threshold()  
    gc.set_threshold(50000, gen1, gen2)
    
    logger.info({
        'INFO': "##GC-COLLECT-TIME##",
        'VALUE': time.time()-start
    })
    
    
def func():
    # 创建调度器BlockingScheduler()
    scheduler = BackgroundScheduler()
    scheduler.add_job(gc_task, 'interval', seconds=TOTAL_INTERVAL, id='auto_gc')
    # 添加任务，时间间隔为5秒
    scheduler.start()