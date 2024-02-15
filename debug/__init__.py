import logging
import sys

print('init')
logging.basicConfig(
    level=logging.INFO,
    filename='debug.log',
    filemode='a',  # 添加这一行，'a'表示append模式，'w'表示write模式
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]  # 添加这一行
)