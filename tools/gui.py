import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import subprocess
import os
import json
import random
import sys
import re
import signal

# GPU Presets: (batch_size, gradient_accumulation, max_length, use_bf16)
GPU_PRESETS = {
    "RTX 5090 (32GB)": (64, 1, 512, True),
    "RTX 4090 (24GB)": (32, 2, 512, True),
    "RTX 3090 (24GB)": (32, 2, 512, True),
    "RTX 4080 (16GB)": (16, 4, 512, True),
    "RTX 3080 (10GB)": (8, 8, 512, False),
    "RTX 4070 (12GB)": (8, 4, 512, True),
}

class TrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Antigravity LFM2-350M Trainer")
        self.root.geometry("1100x850")
        self.root.configure(bg="#121212")

        self.is_windows = os.name == 'nt'

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()

        self.main_frame = tk.Frame(root, bg="#121212", padx=20, pady=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.create_header()
        self.create_settings_frame()
        self.create_dataset_frame()
        self.create_action_buttons()
        self.create_log_frame()

        self.process = None

    def configure_styles(self):
        self.style.configure("TFrame", background="#121212")
        self.style.configure("TLabel", background="#121212", foreground="#e0e0e0", font=("Segoe UI", 10))
        self.style.configure("TCheckbutton", background="#121212", foreground="#e0e0e0")
        
    def create_header(self):
        header_frame = tk.Frame(self.main_frame, bg="#121212")
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(header_frame, text="Liquid LFM2-350M", fg="#00ffcc", bg="#121212", font=("Segoe UI", 24, "bold")).pack()
        tk.Label(header_frame, text="Z-Image Turbo Prompt Fine-Tuner", fg="#888888", bg="#121212", font=("Segoe UI", 10)).pack()

    def create_settings_frame(self):
        settings_container = tk.LabelFrame(self.main_frame, text=" Configuration ", bg="#121212", fg="#00ffcc", font=("Segoe UI", 11, "bold"), padx=15, pady=15, bd=1, relief=tk.SOLID)
        settings_container.pack(fill=tk.X, pady=5)

        # GPU Settings Row
        gpu_row = tk.Frame(settings_container, bg="#121212")
        gpu_row.pack(fill=tk.X)

        tk.Label(gpu_row, text="GPU Preset:", bg="#121212", fg="#e0e0e0").pack(side=tk.LEFT)
        self.preset_var = tk.StringVar(value="RTX 5090 (32GB)")
        self.preset_combo = ttk.Combobox(gpu_row, textvariable=self.preset_var, values=list(GPU_PRESETS.keys()), state="readonly", width=25)
        self.preset_combo.pack(side=tk.LEFT, padx=10)
        self.preset_combo.bind("<<ComboboxSelected>>", self.on_preset_change)

        self.bf16_var = tk.BooleanVar(value=True)
        tk.Checkbutton(gpu_row, text="Enable BF16", variable=self.bf16_var, bg="#121212", fg="#e0e0e0", selectcolor="#2d2d2d", activebackground="#121212", activeforeground="#00ffcc").pack(side=tk.LEFT, padx=20)

        # Hyperparams Row
        hp_row = tk.Frame(settings_container, bg="#121212")
        hp_row.pack(fill=tk.X, pady=10)

        tk.Label(hp_row, text="Steps:", bg="#121212", fg="#e0e0e0").pack(side=tk.LEFT)
        self.steps_var = tk.StringVar(value="3000")
        tk.Entry(hp_row, textvariable=self.steps_var, bg="#2d2d2d", fg="#ffffff", width=10, bd=0).pack(side=tk.LEFT, padx=10)

        tk.Label(hp_row, text="Batch Size:", bg="#121212", fg="#e0e0e0").pack(side=tk.LEFT, padx=(10, 0))
        self.batch_size_var = tk.StringVar(value="64")
        tk.Entry(hp_row, textvariable=self.batch_size_var, bg="#2d2d2d", fg="#ffffff", width=10, bd=0).pack(side=tk.LEFT, padx=10)

        tk.Label(hp_row, text="LR:", bg="#121212", fg="#e0e0e0").pack(side=tk.LEFT, padx=(10, 0))
        self.lr_var = tk.StringVar(value="2e-5")
        tk.Entry(hp_row, textvariable=self.lr_var, bg="#2d2d2d", fg="#ffffff", width=12, bd=0).pack(side=tk.LEFT, padx=10)

    def create_dataset_frame(self):
        ds_container = tk.LabelFrame(self.main_frame, text=" Dataset Source ", bg="#121212", fg="#00ffcc", font=("Segoe UI", 11, "bold"), padx=15, pady=15, bd=1, relief=tk.SOLID)
        ds_container.pack(fill=tk.X, pady=5)

        # File Picker Row
        file_row = tk.Frame(ds_container, bg="#121212")
        file_row.pack(fill=tk.X)

        tk.Label(file_row, text="Training Data (.jsonl):", bg="#121212", fg="#e0e0e0").pack(side=tk.LEFT)
        self.dataset_path_var = tk.StringVar(value="training data/dataset.jsonl")
        self.dataset_entry = tk.Entry(file_row, textvariable=self.dataset_path_var, bg="#2d2d2d", fg="#ffffff", width=60, bd=0)
        self.dataset_entry.pack(side=tk.LEFT, padx=10)
        
        tk.Button(file_row, text="Browse...", command=self.browse_dataset, bg="#444444", fg="#ffffff", bd=0, padx=10).pack(side=tk.LEFT)

        # Data Gen Row
        gen_row = tk.Frame(ds_container, bg="#121212")
        gen_row.pack(fill=tk.X, pady=(15, 0))

        tk.Label(gen_row, text="Synthesize New Prompts:", bg="#121212", fg="#e0e0e0").pack(side=tk.LEFT)
        self.prompt_count_var = tk.StringVar(value="60000")
        tk.Entry(gen_row, textvariable=self.prompt_count_var, bg="#2d2d2d", fg="#ffffff", width=15, bd=0).pack(side=tk.LEFT, padx=10)
        
        self.gen_btn = tk.Button(gen_row, text=" GENERATE & UPDATE ", command=self.run_generation, bg="#007acc", fg="#ffffff", font=("Segoe UI", 9, "bold"), padx=15, bd=0, cursor="hand2")
        self.gen_btn.pack(side=tk.LEFT, padx=5)

    def create_action_buttons(self):
        btn_frame = tk.Frame(self.main_frame, bg="#121212")
        btn_frame.pack(fill=tk.X, pady=20)

        # START BUTTON
        self.train_btn = tk.Button(btn_frame, text=" START TRAINING ", command=self.run_training, bg="#00ffcc", fg="#121212", font=("Segoe UI", 14, "bold"), padx=40, pady=12, bd=0, cursor="hand2")
        self.train_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 10))

        # STOP BUTTON
        self.stop_btn = tk.Button(btn_frame, text=" STOP SESSION ", command=self.stop_process, bg="#ff4444", fg="#ffffff", font=("Segoe UI", 14, "bold"), padx=40, pady=12, bd=0, cursor="hand2", state=tk.DISABLED)
        self.stop_btn.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(10, 0))

    def create_log_frame(self):
        log_container = tk.LabelFrame(self.main_frame, text=" System Output ", bg="#121212", fg="#00ffcc", font=("Segoe UI", 11, "bold"), padx=10, pady=10, bd=1, relief=tk.SOLID)
        log_container.pack(fill=tk.BOTH, expand=True)

        self.log_area = scrolledtext.ScrolledText(log_container, wrap=tk.WORD, bg="#0a0a0a", fg="#00ff00", font=("Consolas", 10), bd=0)
        self.log_area.pack(fill=tk.BOTH, expand=True)

    def browse_dataset(self):
        filename = filedialog.askopenfilename(title="Select Training Data", filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")])
        if filename:
            self.dataset_path_var.set(filename)

    def on_preset_change(self, event=None):
        preset = self.preset_var.get()
        bs, ga, ml, bf = GPU_PRESETS[preset]
        self.batch_size_var.set(str(bs))
        self.bf16_var.set(bf)
        try:
            count = int(self.prompt_count_var.get())
            suggested_steps = (count * 3) // bs
            self.steps_var.set(str(suggested_steps))
        except:
            pass

    def log(self, message):
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)

    def run_generation(self):
        count = self.prompt_count_var.get()
        self.log(f"\n>>> SYNTHESIZING {count} NEW PROMPTS...")
        try:
            with open("tools/generate_data.py", "r") as f:
                content = f.read()
            
            # Use regex or simple replacement for the call
            content = re.sub(r"create_dataset\(\d+\)", f"create_dataset({count})", content)
            content = re.sub(r"count=\d+", f"count={count}", content)
            
            with open("tools/generate_data.py", "w") as f:
                f.write(content)
            
            self.execute_command([sys.executable, "tools/generate_data.py"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update dataset: {e}")

    def run_training(self):
        self.log("\n>>> STARTING TRAINING SESSION...")
        try:
            with open("tools/train.py", "r") as f:
                content = f.read()
            
            content = re.sub(r"max_steps=\d+", f"max_steps={self.steps_var.get()}", content)
            content = re.sub(r"per_device_train_batch_size=\d+", f"per_device_train_batch_size={self.batch_size_var.get()}", content)
            content = re.sub(r"learning_rate=[\d.e-]+", f"learning_rate={self.lr_var.get()}", content)
            content = re.sub(r"bf16=(True|False)", f"bf16={str(self.bf16_var.get())}", content)
            
            # Update Dataset Path
            ds_path = self.dataset_path_var.get().replace("\\", "/")
            content = re.sub(r'DATASET_PATH = ".*?"', f'DATASET_PATH = "{ds_path}"', content)
            
            with open("tools/train.py", "w") as f:
                f.write(content)
            
            self.execute_command([sys.executable, "tools/train.py"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize trainer: {e}")

    def execute_command(self, cmd):
        self.train_btn.config(state=tk.DISABLED, bg="#333333")
        self.gen_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        def run():
            try:
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                
                # Cross-platform subprocess flags
                kwargs = {
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.STDOUT,
                    "text": True,
                    "bufsize": 1,
                    "env": env
                }
                
                if self.is_windows:
                    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                
                self.process = subprocess.Popen(cmd, **kwargs)
                
                for line in self.process.stdout:
                    self.log(line.strip())
                self.process.wait()
            except Exception as e:
                self.log(f"\n>>> ERROR: {e}")
            finally:
                self.root.after(0, self.reset_buttons)

        threading.Thread(target=run, daemon=True).start()

    def reset_buttons(self):
        self.train_btn.config(state=tk.NORMAL, bg="#00ffcc")
        self.gen_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def stop_process(self):
        if self.process:
            if self.is_windows:
                # Group termination on Windows
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.process.pid)], capture_output=True)
            else:
                # Process group termination on Unix
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.log("\n>>> SESSION HALTED BY USER.")

if __name__ == "__main__":
    root = tk.Tk()
    
    # Handle DPI on Windows
    if os.name == 'nt':
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass
    
    app = TrainerGUI(root)
    root.mainloop()
