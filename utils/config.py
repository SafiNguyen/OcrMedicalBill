"""Quản lý cấu hình ứng dụng"""
import yaml
from pathlib import Path

class Config:
    """Singleton Config Manager"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load config từ file yaml"""
        config_path = Path("config.yaml")
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        else:
            raise FileNotFoundError("config.yaml not found!")
    
    def get(self, key_path, default=None):
        """
        Lấy giá trị config theo đường dẫn
        Ví dụ: config.get('tesseract.lang') -> 'vie'
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
