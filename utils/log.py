import atexit
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from colorama import Fore, Style, init
from datetime import datetime
import numpy as np

class CustomLogger:
    def __init__(self, log_name=None, max_bytes=10*1024*1024, backup_count=5):
        # 初始化 colorama
        init(autoreset=True)
        log_file = log_name if log_name else None
        # 配置日志记录器
        self.logger = logging.getLogger(f"custom_logger_{log_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.log_name = log_name
        # 检查是否提供了日志文件路径
        if log_file:
            # 创建一个文件处理器
            self.log_file = log_file
            self.handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8" )
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            self.handler.setFormatter(formatter)
            self.log_to_file = False
        else:
            self.log_file = None
            self.handler = None
            self.log_to_file = False
        

    def __enter__(self):
        # 上下文管理器进入方法
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # 上下文管理器退出方法，记录结束标志（如果启用了文件记录）
        if self.log_to_file:
            self.disable_file_logging()
          

    def enable_file_logging(self):
        # 启用文件记录（仅在提供了日志文件路径时有效）
        if not self.log_to_file and self.handler:
            self._log(f"{self.log_name}日志开始", logging.INFO, Fore.CYAN)
            self.logger.addHandler(self.handler)
            self.log_to_file = True
            self.logger.info("="*10 + f" {self.log_name}日志开始 " + "="*10)
            atexit.register(self.disable_file_logging)

    def disable_file_logging(self):
        # 禁用文件记录
        if self.log_to_file:
            self.logger.info("="*10 + f" {self.log_name}日志结束 " + "="*10)
            self.logger.removeHandler(self.handler)
            self.log_to_file = False
            self._log(f"{self.log_name}日志结束", logging.INFO, Fore.CYAN)

    def log_info(self, message,is_print=True):
        self._log(message, logging.INFO, Fore.GREEN,is_print)

    def log_warning(self, message,is_print=True):
        self._log(message, logging.WARNING, Fore.YELLOW,is_print)

    def log_error(self, message,is_print=True):
        self._log(message, logging.ERROR, Fore.RED,is_print)

    def _log(self, message, level, color,is_print=True):
        # 获取当前时间并格式化
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 序列化消息
        if isinstance(message, (int, float, list, dict, tuple)):
            try:
                message = json.dumps(message, ensure_ascii=False)
            except (TypeError, ValueError):
                message = str(message)
        elif isinstance(message, np.ndarray):
            message = message.tolist()
            message = json.dumps(message, ensure_ascii=False)
        else:
            message = str(message)

        # 打印彩色日志到终端
        if is_print:
            print(f"{color}{current_time} - {message}{Style.RESET_ALL}")
        
        # 记录日志到文件（如果启用了文件记录）
        if self.log_to_file:
            if level == logging.INFO:
                self.logger.info(message)
            elif level == logging.WARNING:
                self.logger.warning(message)
            elif level == logging.ERROR:
                self.logger.error(message)

                
    def get_latest_logs(self):
        # 获取日志文件路径
        if self.log_file and os.path.exists(self.log_file):
            with open(self.log_file, 'r') as file:
                lines = file.readlines()
                if lines:
                    # 找到最后一个 "日志开始" 标志的索引
                    start_index = None
                    for i in range(len(lines) - 1, -1, -1):
                        if "日志开始" in lines[i]:
                            start_index = i
                            break

                    if start_index is not None:
                        # 返回从 "日志开始" 标志到最后的所有日志
                        recent_logs = [self._parse_log(line.strip()) for line in lines[start_index:]]
                        return recent_logs
        return []

    def _parse_log(self, log):
        # 假设日志的格式为 'YYYY-MM-DD HH:MM:SS - MESSAGE'
        try:
            parts = log.split(' - ')
            timestamp_str = parts[0]
            message = parts[1]

            # 尝试将 message 解析为 JSON
            try:
                parsed_message = json.loads(message)
                return parsed_message
            except json.JSONDecodeError:
                # 如果无法解析为 JSON，直接返回字符串
                return message
        except (IndexError, ValueError):
            return log

if __name__ == "__main__":
    # 使用示例
    # 仅打印日志，不记录到文件
    import time
    with CustomLogger() as logger:
        start = time.time()
        logger.log_info("这是一个信息日志（仅终端）")
        logger.log_warning("这是一个警告日志（仅终端）")
        logger.log_error("这是一个错误日志（仅终端）")
        end = time.time()
        print("日志记录耗时:", end-start)

    with CustomLogger('app.log') as logger:
        latest_logs = logger.get_latest_logs()
        print("最新的日志记录:")
        for log in latest_logs:
            print(log)
        logger.enable_file_logging()
        start = time.time()
        logger.log_info(1)
        logger.log_warning(2)
        logger.log_error([3,3])
        logger.log_info(np.array([1,2,3,5,6,7,8,9]))
        end = time.time()
        print("日志记录耗时:", end-start)
        
        
    

       
