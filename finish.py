import sys
import traceback

import psutil
import os
import time
import logging

START_FILE_NAME = 'start.py'


def get_pid(name=None):
    is_python_running = False
    if name is None:
        name = os.path.split(__file__)[-1]
    for proc in psutil.process_iter():
        try:
            if 'python.exe' == proc.name().lower():
                cmd_line = proc.cmdline()
                is_python_running = True
                script_name = None
                for cl in cmd_line:
                    if 'py' == cl.split('.')[-1]:
                        logging.info('target file:' + os.path.split(name)[-1]
                                     + '  current file:' + os.path.split(cl)[-1])
                        scripy_name = os.path.split(cl)[-1]
                        if scripy_name == os.path.split(name)[-1]:
                            return proc.pid, cl
        except psutil.AccessDenied:
            pass
    if not is_python_running:
        logging.info('Python is not running.We can\'t finish any python file.\n')
    return None, None


def kill_process(pid):
    if pid:
        os.system('taskkill /F /IM %s' % pid)
        logging.info('command finished, target file should be closed.\n')


if __name__ == '__main__':
    # 配置日志文件
    logging.basicConfig(level='DEBUG', filename='./finish_logs.txt', filemode='a+')
    logging.info('----------------------------- \n\n'
                 + time.strftime('%y-%m-%d %H:%M:%S') + ' close processing...')
    abs_file_dir = os.path.dirname(__file__)
    start_file_path = os.path.join(abs_file_dir, START_FILE_NAME)
    try:
        if os.path.exists(start_file_path):
            print(start_file_path)
            logging.info('Working dir is:' + start_file_path + '\n')
            pid, script = get_pid(START_FILE_NAME)
            print(pid)
            kill_process(pid)
    except Exception:
        logging.error(time.strftime('%y-%m-%d %H:%M:%S') + traceback.format_exc() + '-------------- \n\n\n\n')
