"""Ứng dụng chính - OCR + KE Đơn Thuốc"""
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from PIL import Image, ImageTk
import threading
from pathlib import Path
import cv2
import numpy as np
import webbrowser
import urllib.parse
import os

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
        self.preprocessing_steps = None  # Store preprocessing images
        
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
        
        # Create frame for result text with hyperlink instructions
        result_info_frame = tk.Frame(tab1, bg="white", height=25)
        result_info_frame.pack(fill="x", padx=5, pady=3)
        tk.Label(result_info_frame, text="💡 Nhấn vào tên thuốc (xanh) để tìm kiếm trên Google",
                font=("Arial", 8), bg="white", fg="#0066CC").pack(side="left")
        
        self.result_text = scrolledtext.ScrolledText(tab1, wrap=tk.WORD, font=("Consolas", 10),
                                                     bg="#F0F4C3", height=20)
        self.result_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Configure hyperlink tag
        self.result_text.tag_configure("drug_link", foreground="#0066CC", underline=True, relief="raised")
        self.result_text.tag_bind("drug_link", "<Button-1>", self.on_drug_click)
        self.result_text.tag_bind("drug_link", "<Enter>", lambda e: self.result_text.config(cursor="hand2"))
        self.result_text.tag_bind("drug_link", "<Leave>", lambda e: self.result_text.config(cursor="arrow"))
        # Also bind a widget-level click to help diagnose clicks that don't hit tag bindings
        self.result_text.bind("<Button-1>", self._result_text_click)
        
        # Store drug names for lookup
        self.drugs_map = {}  # Maps tag names to drug names
        
        tab2 = tk.Frame(tab_control, bg="white")
        tab_control.add(tab2, text="📄 OCR")
        
        self.raw_text = scrolledtext.ScrolledText(tab2, wrap=tk.WORD, font=("Consolas", 9),
                                                  bg="#E8F5E9", height=25)
        self.raw_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # New tab for preprocessing visualization
        tab3 = tk.Frame(tab_control, bg="white")
        tab_control.add(tab3, text="🔍 Tiền xử lý")
        
        self.preprocess_frame = tk.Frame(tab3, bg="white")
        self.preprocess_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.preprocess_label = tk.Label(self.preprocess_frame, text="Chưa có hình ảnh tiền xử lý",
                                        bg="#e0e0e0", relief="sunken")
        self.preprocess_label.pack(fill="both", expand=True)
        
        # Dropdown to select which preprocessing step to view
        control_frame = tk.Frame(tab3, bg="white")
        control_frame.pack(fill="x", padx=5, pady=5)
        
        tk.Label(control_frame, text="Chọn giai đoạn:", bg="white", font=("Arial", 9)).pack(side="left", padx=5)

        self.preprocess_var = tk.StringVar(value="final")
        self.preprocess_dropdown = ttk.Combobox(control_frame, textvariable=self.preprocess_var,
                                               values=["original", "resized", "grayscale", "sharpened",
                                                       "denoised", "binary", "final"],
                                               state="readonly", width=15)
        self.preprocess_dropdown.pack(side="left", padx=5)
        self.preprocess_dropdown.bind("<<ComboboxSelected>>", self.on_preprocessing_step_changed)
        
        tk.Button(control_frame, text="💾 Lưu ảnh", font=("Arial", 9),
                 bg="#FF9800", fg="white", command=self.save_preprocessing_image).pack(side="left", padx=5)
        
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
        self.preprocessing_steps = None
        self.preprocess_label.config(text="Đang xử lý...")
        
        def worker():
            try:
                text, preprocessing_steps = self.ocr_engine.extract_text(
                    self.image_path, 
                    self.update_status,
                    return_preprocessing_steps=True
                )
                
                # Store preprocessing steps for visualization
                self.preprocessing_steps = preprocessing_steps
                
                # Display final preprocessing image
                self.display_preprocessing_step('final')
                
                result = self.keyword_extractor.extract(text, self.update_status)
                result['raw_text'] = text
                
                self.hien_thi_ket_qua(result)
            except Exception as e:
                self.update_status(f"❌ Lỗi: {e}", "red")
            finally:
                self.btn_analyze.config(state="normal")
        
        threading.Thread(target=worker, daemon=True).start()
    
    def hien_thi_ket_qua(self, data):
        """Display results with clickable drug links"""
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.drugs_map = {}  # Reset drug map
        
        # Display patient info
        output = "📝 THÔNG TIN\n" + "="*60 + "\n"
        for info in data['info']:
            output += f"🔹 {info}\n"
        
        self.result_text.insert(tk.END, output)
        
        # Display medications with clickable links
        med_header = "\n💊 THUỐC\n" + "="*60 + "\n"
        self.result_text.insert(tk.END, med_header)
        
        for i, med in enumerate(data['meds'], 1):
            # Extract drug name (first word or phrase before dosage)
            drug_name = self._extract_drug_name(med)
            
            # Insert number and separator
            self.result_text.insert(tk.END, f"{i}. ")
            
            # Create unique tag name for this drug (for lookup)
            tag_name = f"drug_{i}"
            self.drugs_map[tag_name] = drug_name
            
            # Insert drug name with BOTH tags: drug_link (for styling) and drug_i (for data)
            self.result_text.insert(tk.END, drug_name, ("drug_link", tag_name))
            
            # Find the actual position of drug name in original med line and get the rest
            # Use case-insensitive search
            drug_upper = drug_name.upper()
            med_upper = med.upper()
            pos = med_upper.find(drug_upper)
            
            if pos >= 0:
                # Found the drug name in the original text
                rest_start = pos + len(drug_name)
                rest_of_med = med[rest_start:]
            else:
                # Fallback: get everything after first word(s)
                words = med.split()
                # Find first word with dosage unit
                dosage_units = ['mg', 'ml', 'g', 'viên', 'tab', 'lần', 'ngày', 'gói', 'sáng', 'chiều', 'tối']
                rest_start = 0
                for word in words:
                    if any(unit in word.lower() for unit in dosage_units):
                        break
                    rest_start += len(word) + 1
                rest_of_med = med[rest_start:] if rest_start < len(med) else ""
            
            # Insert rest of medication info
            if rest_of_med.strip():
                self.result_text.insert(tk.END, rest_of_med.strip())
            
            self.result_text.insert(tk.END, "\n")
        
        self.result_text.config(state="disabled")
        self.raw_text.insert("1.0", data['raw_text'])
        self.result_data = data
        self.update_status("✅ Hoàn tất!", "green")
    
    def _extract_drug_name(self, medication_line):
        """Extract drug name from medication line"""
        # Known drug names (to match against messy OCR)
        known_drugs = [
            'PROPANOLOL', 'AUGMENTIN', 'AUNGMENTIN', 'PARACETAMOL', 'IBUPROFEN',
            'AMOXICILLIN', 'CEPHALEXIN', 'CEFIXIME', 'OMEPRAZOLE', 'RANITIDINE',
            'SALBUTAMOL', 'LORATADINE', 'CETIRIZINE', 'VITAMIN', 'ASPIRIN',
            'METFORMIN', 'LISINOPRIL', 'AMLODIPINE', 'AEEMUC', 'CETIN'
        ]
        
        med_upper = medication_line.upper()
        
        # Check if line contains any known drug name
        for drug in known_drugs:
            if drug in med_upper:
                # Return the drug name as it appears in config
                if drug == 'AUNGMENTIN':
                    return 'AUGMENTIN'
                return drug
        
        # Fallback: get first word(s) before dosage/instructions
        words = medication_line.split()
        if not words:
            return medication_line
        
        # Find where dosage/instructions start
        dosage_units = ['mg', 'ml', 'g', 'viên', 'tab', 'lần', 'ngày', 'gói', 'sáng', 'chiều', 'tối']
        for i, word in enumerate(words):
            if any(unit in word.lower() for unit in dosage_units):
                # Return everything before the dosage
                if i > 0:
                    return ' '.join(words[:i])
                return words[0]
        
        # If no dosage found, take first word (most likely to be drug name)
        return words[0] if words else medication_line
    
    def on_drug_click(self, event):
        """Handle drug name click - open Google search"""
        print(f"[on_drug_click] Event received: {event}")
        self.logger.debug(f"on_drug_click called with event: {event}")
        
        # Determine index clicked and check tags at that position
        try:
            click_idx = self.result_text.index(f"@{event.x},{event.y}")
            print(f"[on_drug_click] click_idx = {click_idx}")
        except Exception as e:
            print(f"[on_drug_click] Failed to determine click index: {e}")
            self.logger.debug(f"Failed to determine click index: {e}")
            return

        # Log click for debugging
        self.logger.debug(f"Click at index: {click_idx}")
        print(f"[on_drug_click] Click at index: {click_idx}")

        # Directly inspect tags at the clicked index (simpler and more robust)
        tags = self.result_text.tag_names(click_idx)
        self.logger.debug(f"Tags at click: {tags}")
        print(f"[on_drug_click] Tags at click: {tags}")

        for tag in tags:
            print(f"[on_drug_click] Checking tag: {tag}")
            if tag.startswith("drug_"):
                drug_name = self.drugs_map.get(tag)
                print(f"[on_drug_click] Found drug tag {tag}, drug_name = {drug_name}")
                if drug_name:
                    print(f"[on_drug_click] Calling search_drug_google for: {drug_name}")
                    self.logger.info(f"Drug clicked: {drug_name}")
                    # Temporary debug: show confirmation popup so user sees click was received
                    try:
                        messagebox.showinfo("Debug", f"Clicked drug: {drug_name}")
                    except Exception as e:
                        print(f"[on_drug_click] messagebox.showinfo failed: {e}")
                        # If UI popup fails for some reason, keep going
                        pass
                    # Try opening Google in a new browser window/tab (new=2 requests a new window)
                    print(f"[on_drug_click] Calling search_drug_google...")
                    self.search_drug_google(drug_name, new=2)
                    print(f"[on_drug_click] search_drug_google returned")
                    return
                else:
                    print(f"[on_drug_click] drug_name is None/falsy for tag {tag}")
        
        print(f"[on_drug_click] No drug tag found in tags: {tags}")
    
    def search_drug_google(self, drug_name, new=0):
        """Open Google search for drug in default browser.

        Args:
            drug_name: str
            new: int - 0: same browser window, 1: new window, 2: new tab (preferred)
        """
        search_query = f"{drug_name} thuốc"  # Add 'thuốc' for better Vietnamese results
        google_url = f"https://www.google.com/search?q={urllib.parse.quote(search_query)}"

        # Try the normal webbrowser.open first and log the result. If it fails or
        # returns False, fall back to OS-specific handlers (os.startfile on Windows).
        try:
            self.logger.debug(f"Attempting to open URL via webbrowser: {google_url}")
            opened = webbrowser.open(google_url, new=new)
            self.logger.info(f"webbrowser.open returned: {opened}")
            if opened:
                try:
                    messagebox.showinfo("Mở trình duyệt", f"Đã mở trình duyệt để tìm: {drug_name}")
                except Exception:
                    pass
                return
            # If webbrowser.open returned False, attempt fallback
            raise RuntimeError("webbrowser.open returned False")
        except Exception as e:
            self.logger.warning(f"webbrowser.open failed: {e}. Trying OS fallback...")
            # Windows: os.startfile should open the URL in the default browser
            try:
                if os.name == 'nt':
                    os.startfile(google_url)
                else:
                    # Try platform-specific commands for non-Windows
                    import subprocess, sys
                    if sys.platform == 'darwin':
                        subprocess.run(['open', google_url], check=True)
                    else:
                        subprocess.run(['xdg-open', google_url], check=True)

                try:
                    messagebox.showinfo("Mở trình duyệt", f"Đã mở trình duyệt để tìm: {drug_name}")
                except Exception:
                    pass
                return
            except Exception as e2:
                err_msg = f"Không thể mở trình duyệt: {e}; fallback error: {e2}\nURL: {google_url}"
                try:
                    messagebox.showerror("Lỗi", err_msg)
                except Exception:
                    pass
                self.logger.error(err_msg)

    def _result_text_click(self, event):
        """Widget-level click handler used for debugging click/tag issues.

        Shows the text index and tags at the click position and then forwards
        to the normal on_drug_click handler.
        """
        try:
            click_idx = self.result_text.index(f"@{event.x},{event.y}")
            tags = self.result_text.tag_names(click_idx)
            self.logger.debug(f"Widget click at {click_idx}, tags: {tags}")
            # Show a small debug popup so the user can see the click was detected
            try:
                messagebox.showinfo("Debug click", f"Index: {click_idx}\nTags: {tags}")
            except Exception:
                pass
        except Exception as e:
            self.logger.debug(f"Error determining click position: {e}")

        # Forward to the existing tag-based handler
        try:
            self.on_drug_click(event)
        except Exception as e:
            self.logger.error(f"Error in on_drug_click forward: {e}")
    
    def on_preprocessing_step_changed(self, event=None):
        """Handle preprocessing step dropdown change"""
        if not self.preprocessing_steps:
            messagebox.showwarning("Lỗi", "Chưa có hình ảnh tiền xử lý!")
            return
        
        step_name = self.preprocess_var.get()
        if step_name not in self.preprocessing_steps:
            messagebox.showwarning("Lỗi", f"Giai đoạn '{step_name}' không tồn tại!")
            return
        
        self.display_preprocessing_step(step_name)
    
    def display_preprocessing_step(self, step_name):
        """Display a specific preprocessing step"""
        if not self.preprocessing_steps or step_name not in self.preprocessing_steps:
            return
        
        img_array = self.preprocessing_steps[step_name]
        
        # Convert to RGB if grayscale
        if len(img_array.shape) == 2:
            display_img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            display_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Resize if too large
        h, w = display_img.shape[:2]
        if max(h, w) > 400:
            scale = 400 / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            display_img = cv2.resize(display_img, new_size, interpolation=cv2.INTER_AREA)
        
        # Convert to PIL and display
        pil_img = Image.fromarray(display_img)
        photo = ImageTk.PhotoImage(pil_img)
        
        self.preprocess_label.config(image=photo, text="")
        self.preprocess_label.image = photo
    
    def save_preprocessing_image(self):
        """Save current preprocessing step image to file"""
        if not self.preprocessing_steps:
            messagebox.showwarning("Lỗi", "Chưa có hình ảnh tiền xử lý!")
            return
        
        step_name = self.preprocess_var.get()
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")]
        )
        
        if filepath:
            try:
                img_array = self.preprocessing_steps[step_name]
                cv2.imwrite(filepath, img_array)
                messagebox.showinfo("OK", f"Đã lưu: {filepath}")
            except Exception as e:
                messagebox.showerror("Lỗi", str(e))
    
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
