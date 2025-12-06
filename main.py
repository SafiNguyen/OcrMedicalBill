"""Ứng dụng chính - OCR + KE Đơn Thuốc"""
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from PIL import Image, ImageTk
import threading
from pathlib import Path

from utils.config import Config
from utils.logger import Logger
from core.ocr_engine import OCREngine
from core.keyword_extractor import KeywordExtractor

class OCRApp:
    def __init__(self, root):
        self.root = root
        self.config = Config()
        self.logger = Logger.get_logger('OCRApp')
        
        self.root.title(self.config.get('app.name'))
        self.root.geometry(self.config.get('app.window_size'))
        self.root.configure(bg="#f0f0f0")
        
        self.image_path = None
        self.result_data = None
        self.ocr_engine = None
        self.keyword_extractor = None
        
        self.create_widgets()
        self.load_engines()
        
        self.logger.info("App started")
    
    def create_widgets(self):
        # Header
        header = tk.Frame(self.root, bg="#00695C", height=80)
        header.pack(fill="x")
        tk.Label(header, text="🏥 CÔNG CỤ PHÂN TÍCH ĐƠN THUỐC",
                font=("Arial", 18, "bold"), bg="#00695C", fg="white").pack(pady=20)
        
        # Status
        self.status_label = tk.Label(self.root, text="Đang khởi động...",
                                     font=("Arial", 10), bg="#FFF9C4", anchor="w", padx=10)
        self.status_label.pack(fill="x")
        
        # Main
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel
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
        
        # Right panel
        right_panel = tk.Frame(main_frame, bg="white", relief="ridge", bd=2)
        right_panel.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        tk.Label(right_panel, text="📊 KẾT QUẢ", font=("Arial", 12, "bold"),
                bg="white").pack(pady=10)
        
        tab_control = ttk.Notebook(right_panel)
        
        tab1 = tk.Frame(tab_control, bg="white")
        tab_control.add(tab1, text="📝 Kết quả")
        
        self.result_text = scrolledtext.ScrolledText(tab1, wrap=tk.WORD, font=("Consolas", 10),
                                                     bg="#F0F4C3", height=25)
        self.result_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        tab2 = tk.Frame(tab_control, bg="white")
        tab_control.add(tab2, text="📄 OCR")
        
        self.raw_text = scrolledtext.ScrolledText(tab2, wrap=tk.WORD, font=("Consolas", 9),
                                                  bg="#E8F5E9", height=25)
        self.raw_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        tab_control.pack(fill="both", expand=True, padx=5, pady=5)
        
        tk.Button(right_panel, text="💾 Xuất (.txt)", font=("Arial", 10),
                 bg="#FF9800", fg="white", command=self.xuat_ket_qua).pack(pady=5)
    
    def load_engines(self):
        def worker():
            try:
                self.update_status("⏳ Loading OCR...", "orange")
                self.ocr_engine = OCREngine()
                
                self.update_status("⏳ Loading KeyBERT...", "orange")
                self.keyword_extractor = KeywordExtractor()
                
                self.update_status("✅ Sẵn sàng!", "green")
            except Exception as e:
                self.update_status(f"❌ Lỗi: {e}", "red")
        
        threading.Thread(target=worker, daemon=True).start()
    
    def update_status(self, text, color="black"):
        colors = {"green": "#C8E6C9", "orange": "#FFE082", "red": "#FFCDD2"}
        self.status_label.config(text=text, bg=colors.get(color, "#FFF9C4"))
    
    def chon_anh(self):
        filepath = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if filepath:
            self.image_path = filepath
            self.hien_thi_anh(filepath)
            self.btn_analyze.config(state="normal")
            self.update_status("✅ Đã chọn ảnh", "green")
    
    def hien_thi_anh(self, path):
        try:
            img = Image.open(path)
            img.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
        except:
            self.image_label.config(text="Lỗi hiển thị")
    
    def bat_dau_phan_tich(self):
        if not self.image_path:
            messagebox.showwarning("Lỗi", "Chọn ảnh trước!")
            return
        
        if not self.ocr_engine or not self.keyword_extractor:
            messagebox.showwarning("Chờ", "Engine chưa sẵn sàng!")
            return
        
        self.btn_analyze.config(state="disabled")
        self.result_text.delete("1.0", tk.END)
        self.raw_text.delete("1.0", tk.END)
        
        def worker():
            try:
                text = self.ocr_engine.extract_text(self.image_path, self.update_status)
                result = self.keyword_extractor.extract(text, self.update_status)
                result['raw_text'] = text
                
                self.hien_thi_ket_qua(result)
            except Exception as e:
                self.update_status(f"❌ Lỗi: {e}", "red")
            finally:
                self.btn_analyze.config(state="normal")
        
        threading.Thread(target=worker, daemon=True).start()
    
    def hien_thi_ket_qua(self, data):
        output = "📝 THÔNG TIN\n" + "="*60 + "\n"
        for info in data['info']:
            output += f"🔹 {info}\n"
        
        output += "\n💊 THUỐC\n" + "="*60 + "\n"
        for i, med in enumerate(data['meds'], 1):
            output += f"{i}. {med}\n"
        
        self.result_text.insert("1.0", output)
        self.raw_text.insert("1.0", data['raw_text'])
        self.result_data = data
        self.update_status("✅ Hoàn tất!", "green")
    
    def xuat_ket_qua(self):
        if not self.result_data:
            messagebox.showwarning("Lỗi", "Chưa có kết quả!")
            return
        
        filepath = filedialog.asksaveasfilename(defaultextension=".txt",
                                                filetypes=[("Text", "*.txt")])
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(self.result_text.get("1.0", tk.END))
                messagebox.showinfo("OK", f"Đã lưu: {filepath}")
            except Exception as e:
                messagebox.showerror("Lỗi", str(e))

if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()
