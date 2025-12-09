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
    
    def extract_text(self, image_path, callback=None, return_preprocessing_steps=False):
        """Trích xuất văn bản từ ảnh
        
        Args:
            image_path: Path to the image
            callback: Status callback function
            return_preprocessing_steps: If True, returns (text, preprocessing_dict)
            
        Returns:
            If return_preprocessing_steps=False: extracted text string
            If return_preprocessing_steps=True: tuple (text, preprocessing_steps_dict)
        """
        try:
            if callback:
                callback("⏳ Preprocessing image...")
            
            processed_img, preprocessing_steps = self.preprocessor.process(
                image_path, 
                return_steps=True
            )
            
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
            
            if return_preprocessing_steps:
                return text, preprocessing_steps
            return text
            
        except Exception as e:
            self.logger.error(f"OCR error: {e}")
            if callback:
                callback(f"❌ OCR Error: {e}")
            raise
    
    def _post_process_text(self, text):
        """Sửa lỗi OCR phổ biến"""
        # Replace common OCR mistakes
        replacements = {
            'O': '0',  # Letter O to Zero in numbers
            'l': '1',  # Letter l to One in certain contexts
            '|': 'I',  # Pipe to I
        }
        
        # Remove extra spaces and clean up
        text = ' '.join(text.split())
        
        # Remove isolated special characters that are likely noise
        import re
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)  # Remove control chars
        
        return text
