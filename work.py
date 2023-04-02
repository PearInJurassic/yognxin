import sys
import logging
import time
import os
import traceback
import socket

lock_socket_port = 98765

if __name__ == '__main__':
    try:
        file_path = sys.argv[1]
        aim_weight = sys.argv[2]
        logging.basicConfig(level='DEBUG', filename='./logs.txt', filemode='a+')

        lock_socket = socket.socket()
        addr = ('', lock_socket_port)

        try:
            lock_socket.bind(addr)
        except:
            logging.info(time.strftime('%y-%m-%d %H:%M:%S') + 'finish the former start.py' + '-------------- \n\n\n\n')
            os.system('python ./finish.py')
        os.system('start python ./start.py ' + file_path + ' ' + aim_weight)

    except Exception:
        logging.error(time.strftime('%y-%m-%d %H:%M:%S') + traceback.format_exc() + '-------------- \n\n\n\n')
