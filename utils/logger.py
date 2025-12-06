"""Quản lý logging cho ứng dụng"""
import logging
from pathlib import Path
from utils.config import Config

class Logger:
    """Logger Manager"""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name):
        """Lấy logger instance"""
        if name not in cls._loggers:
            cls._loggers[name] = cls._setup_logger(name)
        return cls._loggers[name]
    
    @classmethod
    def _setup_logger(cls, name):
        """Khởi tạo logger"""
        config = Config()
        
        log_dir = Path(config.get('logging.file', 'logs/app.log')).parent
        log_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, config.get('logging.level', 'INFO')))
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(
            config.get('logging.file', 'logs/app.log'),
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            config.get('logging.format', '%(asctime)s - %(levelname)s - %(message)s')
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
