"""Trích xuất từ khóa và phân loại thông tin"""
import re
from utils.config import Config
from utils.logger import Logger

class KeywordExtractor:
    """Keyword Extraction & Classification"""
    
    def __init__(self):
        self.config = Config()
        self.logger = Logger.get_logger('KeywordExtractor')
    
    def extract(self, text, callback=None):
        """Phân tích và phân loại văn bản"""
        try:
            if callback:
                callback("⏳ Analyzing text...")
            
            lines = self._parse_lines(text)
            self.logger.debug(f"Parsed {len(lines)} lines from OCR text")
            
            patient_info, medications = self._classify_lines(lines)
            
            self.logger.info(f"Extracted {len(patient_info)} info lines, {len(medications)} medications")
            
            # If still empty, log the actual text for debugging
            if not patient_info and not medications:
                self.logger.warning(f"No results extracted. Text preview: {text[:200]}")
            
            return {'info': patient_info, 'meds': medications}
            
        except Exception as e:
            self.logger.error(f"Extraction error: {e}")
            raise
    
    def _parse_lines(self, text):
        """Smart line parsing that handles both structured and unstructured OCR text"""
        lines = []
        
        # First try: split by actual line breaks
        potential_lines = text.splitlines()
        
        # If very few lines (< 3), the text is likely one huge block - split it
        if len(potential_lines) < 3:
            # Split on sentence-like boundaries: period + space + capital letter, or common keywords
            segments = re.split(r'(?:\.(?=\s+[A-ZÀÁẢÃẠĂẰẲẴẶÂẦẨẪẬÉÈẺẼẸÊỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỒỔỖỘƠỜỞỠỢÚÙỦŨỤƯỪỬỮỰÝỲỶỸỴĐ])|(?=(?:PROPANOLOL|AUGMENTIN|PARACETAMOL|IBUPROFEN|SỐ|ĐƠN THUỐC)))', text, maxsplit=50)
            potential_lines = [s.strip() for s in segments if s.strip()]
        
        for ln in potential_lines:
            # Clean up whitespace
            s = re.sub(r'\s+', ' ', ln.strip())
            
            # Keep lines that have actual Vietnamese text (not just symbols)
            if len(s) >= 3 and re.search(r'[a-záàảãạăằẳẵặâầẩẫậéèẻẽẹêềểễệíìỉĩịóòỏõọôồổỗộơờởỡợúùủũụưừửữựýỳỷỹỵđA-ZÀÁẢÃẠĂẰẲẴẶÂẦẨẪẬÉÈẺẼẸÊỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỒỔỖỘƠỜỞỠỢÚÙỦŨỤƯỪỬỮỰÝỲỶỸỴĐ]', s):
                lines.append(s)
        
        self.logger.debug(f"Total lines parsed: {len(lines)}")
        if len(lines) < 1:
            self.logger.warning(f"Very few lines parsed. Raw text: {text[:100]}")
        
        return lines
    
    def _classify_lines(self, lines):
        info_keywords = self.config.get('keywords.info', []) or []
        
        patient_info = []
        medications = []
        
        # Build pattern for info keywords with fuzzy matching
        info_pattern_parts = []
        for kw in info_keywords:
            # Make each keyword match even with OCR errors
            info_pattern_parts.append(re.escape(kw.lower()))
        
        if info_pattern_parts:
            info_pattern = re.compile('|'.join(info_pattern_parts), re.I)
        else:
            info_pattern = None
        
        # Medication patterns
        med_pattern = re.compile(r"\d+\s*(mg|ml|g|viên|tab|mcg|%)", re.I)
        dosage_pattern = re.compile(r"\b(uống|sáng|chiều|tối|buổi|lần|ngày|tuần|gói|lần|x)\b", re.I)
        unit_pattern = re.compile(r"\b(mg|ml|g|viên|tab|mcg|%|gói)\b", re.I)
        
        # Common drug names - explicitly list them (including misspelled versions from OCR)
        drug_names = r"\b(PROPANOLOL|AUGMENTIN|AUNGMENTIN|PARACETAMOL|IBUPROFEN|AMOXICILLIN|CEPHALEXIN|CEFIXIME|OMEPRAZOLE|RANITIDINE|SALBUTAMOL|LORATADINE|CETIRIZINE|VITAMIN|ASPIRIN|METFORMIN|LISINOPRIL|AMLODIPINE|LACTASE|SODIUM|CALCIUM|ZINC|IRON|ERYTHROMYCIN|DOXYCYCLINE|FLUOROQUINOLONE)\b"
        drug_pattern = re.compile(drug_names, re.I)
        
        # Exclude patterns - these are definitely NOT medications
        exclude_pattern = re.compile(r"(phòng khám|bệnh viện|bs\.|dr\.|thi|trang|địa|bệnh viện|số điện|quận|thành phố|tỉnh|www|@|\.com|\.vn|^[a-z0-9]{1,2}$)", re.I)
        
        # Clean function
        def clean_for_match(text):
            return re.sub(r'[^a-záàảãạăằẳẵặâầẩẫậéèẻẽẹêềểễệíìỉĩịóòỏõọôồổỗộơờởỡợúùủũụưừửữựýỳỷỹỵđ0-9\s]', '', text.lower())
        
        for ln in lines:
            if not ln or len(ln) < 2:
                continue
            
            lnl = ln.lower()
            lnl_clean = clean_for_match(lnl)
            
            # Skip lines with exclude patterns (but keep if they have drug names)
            if exclude_pattern.search(ln) and not drug_pattern.search(ln):
                continue
            
            # PRIORITY 1: If it contains drug name, it's medication
            if drug_pattern.search(ln):
                medications.append(ln)
                continue
            
            # PRIORITY 2: Check if it contains patient info keyword
            if info_pattern and info_pattern.search(lnl_clean):
                patient_info.append(ln)
                continue
            
            # PRIORITY 3: If it has clear dosage info AND units, likely medication
            if med_pattern.search(ln) and len(ln) < 100:
                medications.append(ln)
                continue
            
            # PRIORITY 4: If it has dosage words AND units, likely medication
            if dosage_pattern.search(lnl_clean) and unit_pattern.search(ln) and len(ln) < 150:
                medications.append(ln)
                continue
            
            # PRIORITY 5: Contains dosage words alone
            if dosage_pattern.search(lnl_clean) and len(ln) < 150:
                medications.append(ln)
        
        # Last resort: return something
        if not medications and not patient_info and lines:
            # Return lines with numbers (likely dosages)
            for ln in lines:
                if re.search(r'\d', ln) and len(ln) < 200:
                    medications.append(ln)
            
            # Or just return first few lines
            if not medications:
                medications = lines[:20]
        
        self.logger.info(f"Extracted {len(patient_info)} info lines, {len(medications)} medications from {len(lines)} parsed lines")
        return patient_info, medications
        
        return patient_info, medications
