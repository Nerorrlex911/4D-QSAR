debug_enabled = True
log_cache = ['Log']
prefix = '[LOG]:'
import datetime

def log(*msgs):
    for msg in msgs:
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        msg = f"{prefix} [{current_time}]: {msg}"
        print(msg)
        log_cache.append(msg)
def save_log(path):
    with open(path, 'w') as f:
        f.write('\n'.join(log_cache))

def debug(msg):
    if debug_enabled:
        print(msg)

def debugfunc(func):
    if debug_enabled:
        func()