import sys
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from PIL import Image, ImageTk
import cv2
import pytesseract
import numpy as np
import re
import threading
import os

# Cấu hình Tesseract (Điều chỉnh đường dẫn phù hợp)
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
    tesseract_path = os.path.join(base_path, 'tesseract', 'tesseract.exe')
else:
    base_path = os.path.dirname(os.path.abspath(__file__))
    tesseract_path = os.path.join(base_path, 'tesseract', 'tesseract.exe')

pytesseract.pytesseract.tesseract_cmd = tesseract_path


# Import KeyBERT
try:
    from keybert import KeyBERT
except ImportError:
    messagebox.showerror("Thiếu thư viện", "Vui lòng chạy: pip install keybert scikit-learn torch")
    sys.exit()


kw_model = None

# ==================================================================
# CẤU HÌNH TỪ KHÓA (GIỐNG CODE CŨ)
# ==================================================================
KEYWORDS_INFO = [
    "họ tên", "họ và tên", "bệnh nhân", "tên bn", "tuổi", "nam sinh", "năm sinh",
    "giới tính", "phái", "mã số", "bhyt", "mã thẻ", "nơi đk", "cân nặng",
    "địa chỉ", "tx.", "tỉnh", "thành phố", "phường", "xã", "huyện",
    "bệnh viện", "pk", "phòng khám", "khoa", "trung tâm",
    "bác sĩ", "bác sỹ", "bs.", "bs,", "ckii", "cki", "ts.bs", "th.s", "người khám",
    "chẩn đoán", "chan doan", "c.đoán", "kèm theo", "bệnh lý",
    "ngày khám", "ngày kê", "ngày cấp", "thời gian", "giá trị từ", "đến ngày"
]

BLACKLIST_TRASH = [
    "stt", "tên thuốc", "hàm lượng", "đvt", "sl", "cách dùng", "ký tên", "ghi rõ",
    "tổng cộng", "cộng khoản", "lời dặn", "tái khám", "mua thêm", "vui lòng", "lưu ý",
    "đơn thuốc", "toa thuốc", "thuốc điều trị", "danh sách thuốc"
]

PATIENT_PATTERNS = {
    "weight": r"\b(\d{1,3})\s*(kg|kgs|kilogram)",
    "height": r"\b(\d{2,3})\s*(cm)",
    "diagnosis": r"(chẩn đoán|chan doan|c\.đoán|bệnh[: ]|icd10|icd-10)",
    "vitals": r"(huyết áp|mạch|spo2|nhiệt độ)",
}

MED_PATTERN = r"^(?:\d{1,2}[\.\)]\s*)?[a-zA-ZÀ-ỹ0-9 ,\-]+?(?:\d+\s*(mg|ml|g|mcg|viên|vỉ|ống|chai|gói))"
DOSAGE_PATTERN = r"(uống|sáng|chiều|tối|trưa|lần|viên|ngày|chia)"



MED_UNITS = r"\d+\s*(mg|ml|g|mcg|viên|ống|chai|tuýp|gói|vỉ|cap|tab|lần)"
DOSAGE_KEYWORDS = ["uống", "sáng", "chiều", "tối", "trưa", "ăn", "thoa", "xịt", "nhỏ", "tiêm", "chia"]

# ==================================================================
# HÀM TIỀN XỬ LÝ ẢNH (ADAPTIVE THRESHOLD)
# ==================================================================
def xu_ly_anh(image_path):
    """Tiền xử lý ảnh trước khi OCR"""
    try:
        stream = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # sharpen
        kernel_sharp = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]])
        sharp = cv2.filter2D(gray, -1, kernel_sharp)

        # reduce noise
        denoise = cv2.fastNlMeansDenoising(sharp, h=12)

        # adaptive threshold
        binary = cv2.adaptiveThreshold(
            denoise, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 7
        )
        return binary
    except Exception as e:
        raise Exception(f"Lỗi xử lý ảnh: {e}")

# ==================================================================
# HÀM PHÂN TÍCH ĐƠN THUỐC (KẾT HỢP OCR + KE)
# ==================================================================
def phan_tich_don_thuoc(image_path, callback):
    """Xử lý OCR + KeyBERT để phân tách thông tin"""
    try:
        # Bước 1: OCR
        callback("⏳ Đang tiền xử lý ảnh...")
        processed_img = xu_ly_anh(image_path)
        
        callback("⏳ Đang quét văn bản (OCR)...")
        text = pytesseract.image_to_string(processed_img, lang='vie', config='--oem 3 --psm 6 -c preserve_interword_spaces=1')
        
        if not text.strip():
            callback("❌ Không đọc được văn bản từ ảnh!")
            return None
        
        # Bước 2: KeyBERT phân tích
        callback("⏳ Đang phân tích từ khóa (KeyBERT)...")
        
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        patient_info = []
        meds = []

        for raw in lines:
            lower = raw.lower()

            # 1. patient info
            if any(k in lower for k in KEYWORDS_INFO):
                patient_info.append(raw)
                continue

            for key, pattern in PATIENT_PATTERNS.items():
                if re.search(pattern, lower):
                    patient_info.append(raw)
                    break
            else:
                # 2. thuốc
                if re.search(MED_PATTERN, lower) or re.search(DOSAGE_PATTERN, lower):
                    meds.append(raw)

        return {
            "info": patient_info,
            "meds": meds,
            "raw_text": text
        }

    except Exception as e:
        callback(f"❌ Lỗi: {e}")
        return None

# ==================================================================
# GIAO DIỆN ỨNG DỤNG
# ==================================================================
class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR + KE - ĐƠN THUỐC VIỆT NAM")
        self.root.geometry("1200x750")
        self.root.configure(bg="#f0f0f0")
        
        self.image_path = None
        self.result_data = None
        
        self.create_widgets()
        self.load_model()
    
    def create_widgets(self):
        # Header
        header = tk.Frame(self.root, bg="#00695C", height=80)
        header.pack(fill="x")
        
        tk.Label(header, text="🏥 CÔNG CỤ PHÂN TÍCH ĐƠN THUỐC", 
                font=("Arial", 18, "bold"), bg="#00695C", fg="white").pack(pady=20)
        
        # Status bar
        self.status_label = tk.Label(self.root, text="Đang khởi động...", 
                                     font=("Arial", 10), bg="#FFF9C4", fg="#333", anchor="w", padx=10)
        self.status_label.pack(fill="x")
        
        # Main content
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel - Image
        left_panel = tk.Frame(main_frame, bg="white", relief="ridge", bd=2)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        tk.Label(left_panel, text="📷 ẢNH ĐƠN THUỐC", font=("Arial", 12, "bold"), 
                bg="white").pack(pady=10)
        
        self.image_label = tk.Label(left_panel, text="Chưa có ảnh", bg="#e0e0e0", 
                                    width=40, height=20, relief="sunken")
        self.image_label.pack(padx=10, pady=10, fill="both", expand=True)
        
        btn_frame = tk.Frame(left_panel, bg="white")
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="📁 Chọn ảnh", font=("Arial", 11), 
                 bg="#2196F3", fg="white", padx=15, command=self.chon_anh).pack(side="left", padx=5)
        
        self.btn_analyze = tk.Button(btn_frame, text="🚀 PHÂN TÍCH", font=("Arial", 11, "bold"),
                                     bg="#4CAF50", fg="white", padx=20, state="disabled",
                                     command=self.bat_dau_phan_tich)
        self.btn_analyze.pack(side="left", padx=5)
        
        # Right panel - Results
        right_panel = tk.Frame(main_frame, bg="white", relief="ridge", bd=2)
        right_panel.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        tk.Label(right_panel, text="📊 KẾT QUẢ PHÂN TÍCH", font=("Arial", 12, "bold"),
                bg="white").pack(pady=10)
        
        # Tabs
        tab_control = ttk.Notebook(right_panel)
        
        # Tab 1: Kết quả phân loại
        tab1 = tk.Frame(tab_control, bg="white")
        tab_control.add(tab1, text="📝 Thông tin & Thuốc")
        
        self.result_text = scrolledtext.ScrolledText(tab1, wrap=tk.WORD, font=("Consolas", 10),
                                                     bg="#F0F4C3", height=25)
        self.result_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Tab 2: Văn bản gốc
        tab2 = tk.Frame(tab_control, bg="white")
        tab_control.add(tab2, text="📄 Văn bản OCR")
        
        self.raw_text = scrolledtext.ScrolledText(tab2, wrap=tk.WORD, font=("Consolas", 9),
                                                  bg="#E8F5E9", height=25)
        self.raw_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        tab_control.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Export button
        tk.Button(right_panel, text="💾 Xuất kết quả (.txt)", font=("Arial", 10),
                 bg="#FF9800", fg="white", command=self.xuat_ket_qua).pack(pady=5)
    
    def load_model(self):
        """Tải KeyBERT model trong background"""
        def tai():
            global kw_model
            try:
                self.update_status("⏳ Đang tải KeyBERT model... (30s)", "orange")
                kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
                self.update_status("✅ Sẵn sàng! Hãy chọn ảnh đơn thuốc.", "green")
            except Exception as e:
                self.update_status(f"❌ Lỗi tải model: {e}", "red")
        
        threading.Thread(target=tai, daemon=True).start()
    
    def update_status(self, text, color="black"):
        """Cập nhật status bar"""
        if color == "green":
            bg = "#C8E6C9"
        elif color == "orange":
            bg = "#FFE082"
        elif color == "red":
            bg = "#FFCDD2"
        else:
            bg = "#FFF9C4"
        
        self.status_label.config(text=text, bg=bg)
    
    def chon_anh(self):
        """Chọn ảnh đơn thuốc"""
        filepath = filedialog.askopenfilename(
            title="Chọn ảnh đơn thuốc",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if filepath:
            self.image_path = filepath
            self.hien_thi_anh(filepath)
            self.btn_analyze.config(state="normal")
            self.update_status("✅ Đã chọn ảnh. Nhấn 'PHÂN TÍCH' để bắt đầu.", "green")
    
    def hien_thi_anh(self, path):
        """Hiển thị ảnh preview"""
        try:
            img = Image.open(path)
            img.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
        except:
            self.image_label.config(text="Không thể hiển thị ảnh")
    
    def bat_dau_phan_tich(self):
        """Bắt đầu phân tích trong thread riêng"""
        if not self.image_path:
            messagebox.showwarning("Lỗi", "Vui lòng chọn ảnh trước!")
            return
        
        if kw_model is None:
            messagebox.showwarning("Chờ", "Model chưa tải xong!")
            return
        
        self.btn_analyze.config(state="disabled")
        self.result_text.delete("1.0", tk.END)
        self.raw_text.delete("1.0", tk.END)
        
        def xu_ly():
            result = phan_tich_don_thuoc(self.image_path, self.update_status)
            if result:
                self.hien_thi_ket_qua(result)
            self.btn_analyze.config(state="normal")
        
        threading.Thread(target=xu_ly, daemon=True).start()
    
    def hien_thi_ket_qua(self, data):
        """Hiển thị kết quả lên giao diện"""
        # Tab 1: Kết quả phân loại
        output = ""
        output += "📝 MỤC 1: THÔNG TIN CHUNG\n"
        output += "=" * 50 + "\n"
        if data['info']:
            for info in data['info']:
                output += f"🔹 {info}\n"
        else:
            output += "(Không tìm thấy)\n"
        
        output += "\n💊 MỤC 2: DANH SÁCH THUỐC\n"
        output += "=" * 50 + "\n"
        if data['meds']:
            for i, med in enumerate(data['meds'], 1):
                output += f"{i}. {med}\n"
        else:
            output += "(Không tìm thấy)\n"
        
        self.result_text.insert("1.0", output)
        
        # Tab 2: Văn bản gốc
        self.raw_text.insert("1.0", data['raw_text'])
        
        self.result_data = data
        self.update_status("✅ Phân tích hoàn tất!", "green")
    
    def xuat_ket_qua(self):
        """Xuất kết quả ra file .txt"""
        if not self.result_data:
            messagebox.showwarning("Lỗi", "Chưa có kết quả để xuất!")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            title="Lưu kết quả"
        )
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(self.result_text.get("1.0", tk.END))
                messagebox.showinfo("Thành công", f"Đã lưu: {filepath}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu file: {e}")

# ==================================================================
# KHỞI CHẠY ỨNG DỤNG
# ==================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()