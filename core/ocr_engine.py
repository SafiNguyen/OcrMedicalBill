"""OCR Engine sử dụng Tesseract"""
import sys
import os
import pytesseract
from pathlib import Path
from utils.config import Config
from utils.logger import Logger
from core.preprocessor import ImagePreprocessor

class OCREngine:
    """Tesseract OCR Engine"""
    
    def __init__(self):
        self.config = Config()
        self.logger = Logger.get_logger('OCREngine')
        self.preprocessor = ImagePreprocessor()
        self._setup_tesseract()
    
    def _setup_tesseract(self):
        """Cấu hình đường dẫn Tesseract"""
        try:
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS
            else:
                base_path = Path(__file__).parent.parent
            
            tesseract_path = os.path.join(
                base_path,
                self.config.get('tesseract.path', 'tesseract/tesseract.exe')
            )
            
            if not os.path.exists(tesseract_path):
                tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            self.logger.info(f"Tesseract path: {tesseract_path}")
            
        except Exception as e:
            self.logger.error(f"Tesseract setup error: {e}")
            raise
    
    def extract_text(self, image_path, callback=None):
        """Trích xuất văn bản từ ảnh"""
        try:
            if callback:
                callback("⏳ Preprocessing image...")
            
            processed_img = self.preprocessor.process(image_path)
            
            if callback:
                callback("⏳ Running OCR...")
            
            lang = self.config.get('tesseract.lang', 'vie')
            config_str = self.config.get('tesseract.config', '--oem 1 --psm 6')
            
            text = pytesseract.image_to_string(
                processed_img,
                lang=lang,
                config=config_str
            )
            
            text = self._post_process_text(text)
            
            self.logger.info(f"OCR completed. Extracted {len(text)} characters")
            return text
            
        except Exception as e:
            self.logger.error(f"OCR error: {e}")
            if callback:
                callback(f"❌ OCR Error: {e}")
            raise
    
    def _post_process_text(self, text):
        """Sửa lỗi OCR phổ biến"""
        fixes = {
            'độ go}': 'Địa chỉ',
            'Họiên': 'Họ tên',
            'Namj': 'Nam',
            'Điệnthoại': 'Điện thoại',
            'Chẳn đoán': 'Chẩn đoán',
            'sảng': 'sáng',
            'Ngày ti': 'Ngày tái',
        }
        
        for wrong, correct in fixes.items():
            text = text.replace(wrong, correct)
        
        return text
