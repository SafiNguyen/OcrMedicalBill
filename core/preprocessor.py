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
        self.processing_steps = {}  # Store intermediate images
    
    def process(self, image_path, return_steps=False):
        """Tiền xử lý ảnh
        
        Args:
            image_path: Path to the image
            return_steps: If True, returns dict with intermediate processing stages
            
        Returns:
            If return_steps=False: processed image (final cleaned image)
            If return_steps=True: tuple (processed_image, steps_dict)
        """
        try:
            self.logger.info(f"Processing image: {image_path}")
            self.processing_steps = {}
            
            img = self._read_image(image_path)
            self.processing_steps['original'] = img.copy()
            
            img = self._resize_if_needed(img)
            self.processing_steps['resized'] = img.copy()
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.processing_steps['grayscale'] = gray.copy()
            
            sharp = self._sharpen_image(gray)
            self.processing_steps['sharpened'] = sharp.copy()
            
            denoised = self._denoise_image(sharp)
            self.processing_steps['denoised'] = denoised.copy()

            # Apply binary adaptive threshold to enhance text contrast
            binary = self._apply_threshold(denoised)
            self.processing_steps['binary'] = binary.copy()

            # Morphological cleaning on the binary image to produce final output
            clean = self._morphological_clean(binary)
            self.processing_steps['final'] = clean.copy()
            
            self.logger.info("Image preprocessing completed")
            
            if return_steps:
                return clean, self.processing_steps
            return clean
            
        except Exception as e:
            self.logger.error(f"Preprocessing error: {e}")
            raise
    
    def get_processing_steps(self):
        """Get the intermediate processing steps"""
        return self.processing_steps
    
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
        # Use unsharp mask for better sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(gray, -1, kernel)
    
    def _denoise_image(self, img):
        """Khử nhiễu"""
        h = self.config.get('ocr.denoise_strength', 10)
        h = int(h) if h else 10
        return cv2.fastNlMeansDenoising(
            img, 
            h=h, 
            templateWindowSize=7, 
            searchWindowSize=21
        )
    
    def _apply_threshold(self, img):
        """Adaptive threshold"""
        block_size = self.config.get('ocr.adaptive_threshold_block', 31)
        c = self.config.get('ocr.adaptive_threshold_c', 9)
        
        block_size = int(block_size) if block_size else 31
        c = int(c) if c else 9
        
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
        
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
