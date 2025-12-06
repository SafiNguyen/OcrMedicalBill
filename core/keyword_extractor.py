"""Trích xuất từ khóa và phân loại thông tin"""
import re
from utils.config import Config
from utils.logger import Logger

class KeywordExtractor:
    """Keyword Extraction & Classification"""
    
    def __init__(self):
        self.config = Config()
        self.logger = Logger.get_logger('KeywordExtractor')
        self.kw_model = None
        self._load_keybert()
    
    def _load_keybert(self):
        """Load KeyBERT model"""
        try:
            from keybert import KeyBERT
            model_name = self.config.get('models.keybert')
            self.kw_model = KeyBERT(model=model_name)
            self.logger.info("KeyBERT loaded")
        except Exception as e:
            self.logger.warning(f"KeyBERT not loaded: {e}")
            self.kw_model = None
    
    def extract(self, text, callback=None):
        """Phân tích và phân loại văn bản"""
        try:
            if callback:
                callback("⏳ Analyzing text...")
            
            keybert_words = self._get_keybert_keywords(text)
            lines = self._parse_lines(text)
            patient_info, medications = self._classify_lines(lines, keybert_words)
            
            return {'info': patient_info, 'meds': medications}
            
        except Exception as e:
            self.logger.error(f"Extraction error: {e}")
            raise
    
    def _get_keybert_keywords(self, text):
        if self.kw_model is None:
            return []
        try:
            kws = self.kw_model.extract_keywords(text, keyphrase_ngram_range=(1,2), top_n=40)
            return [k[0].lower() for k in kws if isinstance(k, (list, tuple)) and k]
        except:
            return []
    
    def _parse_lines(self, text):
        lines = []
        for ln in text.splitlines():
            s = re.sub(r'\s+', ' ', ln.strip())
            if len(s) >= 2:
                lines.append(s)
        return lines
    
    def _classify_lines(self, lines, keybert_words):
        info_keywords = self.config.get('keywords.info', [])
        blacklist = self.config.get('keywords.blacklist', [])
        
        patient_info = []
        medications = []
        
        med_pattern = re.compile(r"\d+\s*(mg|ml|g|viên|tab)", re.I)
        dosage_pattern = re.compile(r"\b(uống|sáng|chiều|tối)\b", re.I)
        
        for ln in lines:
            lnl = ln.lower()
            
            if any(bad in lnl for bad in blacklist):
                continue
            
            if any(k in lnl for k in info_keywords):
                patient_info.append(ln)
                continue
            
            if dosage_pattern.search(lnl) or med_pattern.search(ln):
                medications.append(ln)
        
        return patient_info, medications
