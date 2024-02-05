import logging

print('init')
logging.basicConfig(
    level=logging.INFO,
    filename='debug.log',
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)