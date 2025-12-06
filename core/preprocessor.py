"""Xử lý tiền xử lý ảnh trước OCR"""
import cv2
import numpy as np
from utils.config import Config
from utils.logger import Logger

class ImagePreprocessor:
    """Image Preprocessing Pipeline"""
    
    def __init__(self):
        self.config = Config()
        self.logger = Logger.get_logger('ImagePreprocessor')
    
    def process(self, image_path):
        """Tiền xử lý ảnh"""
        try:
            self.logger.info(f"Processing image: {image_path}")
            
            img = self._read_image(image_path)
            img = self._resize_if_needed(img)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sharp = self._sharpen_image(gray)
            denoised = self._denoise_image(sharp)
            binary = self._apply_threshold(denoised)
            clean = self._morphological_clean(binary)
            
            self.logger.info("Image preprocessing completed")
            return clean
            
        except Exception as e:
            self.logger.error(f"Preprocessing error: {e}")
            raise
    
    def _read_image(self, path):
        """Đọc ảnh (hỗ trợ Unicode path)"""
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Cannot read image")
        
        return img
    
    def _resize_if_needed(self, img):
        """Resize ảnh nếu quá lớn"""
        max_dim = self.config.get('ocr.max_image_dimension', 1600)
        h, w = img.shape[:2]
        
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            self.logger.debug(f"Resized image to {new_size}")
        
        return img
    
    def _sharpen_image(self, gray):
        """Sharpen để tăng độ nét chữ"""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(gray, -1, kernel)
    
    def _denoise_image(self, img):
        """Khử nhiễu"""
        h = self.config.get('ocr.denoise_strength', 10)
        return cv2.fastNlMeansDenoising(img, h=h, templateWindowSize=7, searchWindowSize=21)
    
    def _apply_threshold(self, img):
        """Adaptive threshold"""
        block_size = self.config.get('ocr.adaptive_threshold_block', 31)
        c = self.config.get('ocr.adaptive_threshold_c', 9)
        
        return cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size, c
        )
    
    def _morphological_clean(self, img):
        """Morphological cleaning"""
        kernel = np.ones((1, 1), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
