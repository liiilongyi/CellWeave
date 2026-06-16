import os
import sys
import json
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

for folder in ["curve", "figure", "stardist_npy", os.path.join("weight", "MoE")]:
    os.makedirs(folder, exist_ok=True)

class PipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multimodal Single-Cell Spatial Transcriptomics Segmentation Pipeline (GNN-MoE)")
        self.root.geometry("850x700")

        # ==================== Core Modification: Streamlined config dictionary and updated the default path for IMAGE_PATH ====================
        self.config = {
            "SAMPLE_ID": "mouse_brain",
            "IMAGE_PATH": "UNI/mouse_brain_demo.png",  # <--- Updated to real physical path
            "POS_PATH": "binned_outputs/square_002um/spatial/tissue_positions.parquet",
            "EXPR_FILTERED_H5": "binned_outputs/square_002um/filtered_feature_bc_matrix.h5",
            "EXPR_RAW_H5": "binned_outputs/square_002um/raw_feature_bc_matrix.h5"
        }
        # =====================================================================================

        self.running_process = None
        self.create_widgets()
        self.load_config()

    def create_widgets(self):
        # 1. Path Configuration Area
        path_frame = tk.LabelFrame(self.root, text="Project Path Configuration", padx=10, pady=10)
        path_frame.pack(fill="x", padx=15, pady=5)

        # Dataset / Sample ID Configuration
        tk.Label(path_frame, text="Dataset/Sample ID (SAMPLE_ID):").grid(row=0, column=0, sticky="w")
        self.entry_id = tk.Entry(path_frame, width=50)
        self.entry_id.grid(row=0, column=1, padx=5, pady=2)

        # H&E Image Path
        tk.Label(path_frame, text="H&E Image (.png):").grid(row=1, column=0, sticky="w")
        self.entry_img = tk.Entry(path_frame, width=50, state="readonly")
        self.entry_img.grid(row=1, column=1, padx=5, pady=2)
        tk.Button(path_frame, text="Browse",
                  command=lambda: self.select_file(self.entry_img, "IMAGE_PATH", [("PNG Image", "*.png")])).grid(row=1,
                                                                                                                 column=2,
                                                                                                                 padx=5,
                                                                                                                 pady=2)

        # Tissue Positions Path
        tk.Label(path_frame, text="Tissue Positions (.parquet):").grid(row=2, column=0, sticky="w")
        self.entry_pos = tk.Entry(path_frame, width=50, state="readonly")
        self.entry_pos.grid(row=2, column=1, padx=5, pady=2)
        tk.Button(path_frame, text="Browse",
                  command=lambda: self.select_file(self.entry_pos, "POS_PATH", [("Parquet File", "*.parquet")])).grid(
            row=2, column=2, padx=5, pady=2)

        # Filtered Expression Matrix (.h5)
        tk.Label(path_frame, text="Filtered Expression (.h5):").grid(row=3, column=0, sticky="w")
        self.entry_filtered_h5 = tk.Entry(path_frame, width=50, state="readonly")
        self.entry_filtered_h5.grid(row=3, column=1, padx=5, pady=2)
        tk.Button(path_frame, text="Browse",
                  command=lambda: self.select_file(self.entry_filtered_h5, "EXPR_FILTERED_H5",
                                                   [("h5 File", "*.h5")])).grid(row=3, column=2, padx=5, pady=2)

        # Unfiltered Raw Expression Matrix (.h5)
        tk.Label(path_frame, text="Unfiltered Raw Expression (.h5):").grid(row=4, column=0, sticky="w")
        self.entry_raw_h5 = tk.Entry(path_frame, width=50, state="readonly")
        self.entry_raw_h5.grid(row=4, column=1, padx=5, pady=2)
        tk.Button(path_frame, text="Browse",
                  command=lambda: self.select_file(self.entry_raw_h5, "EXPR_RAW_H5", [("h5 File", "*.h5")])).grid(row=4,
                                                                                                                  column=2,
                                                                                                                  padx=5,
                                                                                                                  pady=2)

        # 2. Control Area
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill="x", padx=15, pady=10)

        # Step-by-step execution buttons (7 steps)
        btn_width = 14
        tk.Button(control_frame, text="1.Preprocess", width=btn_width, bg="#E1F5FE",
                  command=lambda: self.start_step(1)).pack(side="left", padx=4)
        tk.Button(control_frame, text="2.scGPT Encode", width=btn_width, bg="#E1F5FE",
                  command=lambda: self.start_step(2)).pack(side="left", padx=4)
        tk.Button(control_frame, text="3.UNI Features", width=btn_width, bg="#E1F5FE",
                  command=lambda: self.start_step(3)).pack(side="left", padx=4)
        tk.Button(control_frame, text="4.StarDist Seg", width=btn_width, bg="#E1F5FE",
                  command=lambda: self.start_step(4)).pack(side="left", padx=4)
        tk.Button(control_frame, text="5.Seeds&Labels", width=btn_width, bg="#E1F5FE",
                  command=lambda: self.start_step(5)).pack(side="left", padx=4)
        tk.Button(control_frame, text="6.GNN Training", width=btn_width, bg="#FFF9C4",
                  command=lambda: self.start_step(6)).pack(side="left", padx=4)
        tk.Button(control_frame, text="7.Cell Diffusion", width=btn_width, bg="#C8E6C9",
                  command=lambda: self.start_step(7)).pack(side="left", padx=4)

        # Full Automatic Execution
        action_frame = tk.Frame(self.root)
        action_frame.pack(fill="x", padx=15, pady=5)
        self.btn_run_all = tk.Button(action_frame, text="🚀 Run Entire Pipeline Automatically", bg="#4CAF50", fg="white",
                                     font=("Arial", 10, "bold"), pady=5, command=self.run_all)
        self.btn_run_all.pack(side="left", fill="x", expand=True, padx=5)

        self.btn_stop = tk.Button(action_frame, text="🛑 Stop Current Process", bg="#F44336", fg="white",
                                  font=("Arial", 10, "bold"), pady=5, state="disabled", command=self.stop_process)
        self.btn_stop.pack(side="right", padx=5)

        # Progress Bar
        self.progress = ttk.Progressbar(self.root, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", padx=15, pady=5)

        # 3. Log Console
        log_frame = tk.LabelFrame(self.root, text="Real-time Execution Logs")
        log_frame.pack(fill="both", expand=True, padx=15, pady=10)
        self.log_area = ScrolledText(log_frame, wrap="char", bg="black", fg="#00FF00", font=("Consolas", 9))
        self.log_area.pack(fill="both", expand=True, padx=5, pady=5)

    def select_file(self, entry, config_key, filetypes):
        """Browse file: Save the full relative path in memory, display only the filename in the entry box"""
        file = filedialog.askopenfilename(filetypes=filetypes)
        if file:
            rel_path = os.path.relpath(file, os.getcwd())
            # 1. Save full relative path to memory config to ensure correct computation
            self.config[config_key] = rel_path

            # 2. Enable, update, and reset to readonly on UI to display the clean filename
            entry.config(state="normal")
            entry.delete(0, tk.END)
            entry.insert(0, os.path.basename(rel_path))
            entry.config(state="readonly")

    def write_log(self, text):
        self.log_area.insert(tk.END, text)
        self.log_area.see(tk.END)

    def save_config(self):
        # Allow manual input of SAMPLE_ID
        self.config["SAMPLE_ID"] = self.entry_id.get()

        # ==================== Core Modification: Keep only 5 core input paths when writing to disk ====================
        core_config = {
            "SAMPLE_ID": self.config.get("SAMPLE_ID", "mouse_brain"),
            "IMAGE_PATH": self.config.get("IMAGE_PATH", ""),
            "POS_PATH": self.config.get("POS_PATH", ""),
            "EXPR_FILTERED_H5": self.config.get("EXPR_FILTERED_H5", ""),
            "EXPR_RAW_H5": self.config.get("EXPR_RAW_H5", "")
        }
        self.config = core_config  # Update memory config

        with open("pipeline_config.json", "w", encoding="utf-8") as f:
            json.dump(core_config, f, indent=4)
        self.write_log("[SYSTEM] Configuration file pipeline_config.json updated and saved.\n")
        # ===================================================================================

    def load_config(self):
        if os.path.exists("pipeline_config.json"):
            try:
                with open("pipeline_config.json", "r", encoding="utf-8") as f:
                    # Read latest configuration
                    loaded_cfg = json.load(f)
                    # Filter out extra keys to ensure consistency of core fields
                    self.config = {k: loaded_cfg.get(k, self.config[k]) for k in self.config.keys()}
            except Exception as e:
                print(f"Failed to load configuration: {e}")

        # 1. Load Sample ID
        self.entry_id.delete(0, tk.END)
        self.entry_id.insert(0, self.config.get("SAMPLE_ID", "mouse_brain"))

        # 2. Load image path (only display short filename, keep readonly)
        self.entry_img.config(state="normal")
        self.entry_img.delete(0, tk.END)
        self.entry_img.insert(0, os.path.basename(self.config.get("IMAGE_PATH", "UNI/mouse_brain_demo.png")))
        self.entry_img.config(state="readonly")

        # 3. Load tissue position path (only display short filename, keep readonly)
        self.entry_pos.config(state="normal")
        self.entry_pos.delete(0, tk.END)
        self.entry_pos.insert(0, os.path.basename(self.config.get("POS_PATH", "tissue_positions.parquet")))
        self.entry_pos.config(state="readonly")

        # 4. Load filtered H5 path (only display short filename, keep readonly)
        self.entry_filtered_h5.config(state="normal")
        self.entry_filtered_h5.delete(0, tk.END)
        self.entry_filtered_h5.insert(0, os.path.basename(
            self.config.get("EXPR_FILTERED_H5", "filtered_feature_bc_matrix.h5")))
        self.entry_filtered_h5.config(state="readonly")

        # 5. Load raw H5 path (only display short filename, keep readonly)
        self.entry_raw_h5.config(state="normal")
        self.entry_raw_h5.delete(0, tk.END)
        self.entry_raw_h5.insert(0, os.path.basename(self.config.get("EXPR_RAW_H5", "raw_feature_bc_matrix.h5")))
        self.entry_raw_h5.config(state="readonly")

    def stop_process(self):
        if self.running_process:
            self.running_process.terminate()
            self.write_log("\n⚠️ Process interrupted manually by user!\n")
            self.btn_stop.config(state="disabled")
            self.btn_run_all.config(state="normal")

    def run_cmd(self, cmd, cwd=None, throttle=False):
        """Core wrapper: Execute each Python script in a non-blocking real-time streaming manner"""
        self.write_log(f"\n[RUNNING] >>> Executing command: {cmd}\n")

        sub_env = os.environ.copy()
        sub_env["PYTHONIOENCODING"] = "utf-8"

        self.running_process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=True,
            encoding="utf-8",
            errors="ignore",
            env=sub_env
        )

        self.btn_stop.config(state="normal")
        line_count = 0

        while True:
            output = self.running_process.stdout.readline()
            if output == '' and self.running_process.poll() is not None:
                break
            if output:
                if throttle:
                    lower_out = output.lower()
                    is_critical = any(kw in lower_out for kw in [
                        "error", "traceback", "exception", "failed",
                        "warning", "warn", "success", "done", "matching", "filtering"
                    ]) or any(emoji in output for f, emoji in [("ok", "✅"), ("err", "❌"), ("warn", "⚠️")])

                    if is_critical:
                        self.write_log(output)
                    else:
                        line_count += 1
                        if line_count >= 100:
                            self.write_log(output)
                            line_count = 0
                else:
                    self.write_log(output)

        rc = self.running_process.poll()
        self.btn_stop.config(state="disabled")
        return rc == 0

    def start_step(self, step_idx):
        self.save_config()
        self.btn_run_all.config(state="disabled")
        threading.Thread(target=self.run_pipeline, args=([step_idx],), daemon=True).start()

    def run_all(self):
        self.save_config()
        self.btn_run_all.config(state="disabled")
        threading.Thread(target=self.run_pipeline, args=([1, 2, 3, 4, 5, 6, 7],), daemon=True).start()

    def run_pipeline(self, steps):
        total_steps = len(steps)
        current_progress = 0
        self.progress["value"] = 0

        if getattr(sys, 'frozen', False):
            python_exe = r"D:\xuehao\Anaconda\envs\cellweave\python.exe"
        else:
            python_exe = sys.executable

        for step in steps:
            success = False
            if step == 1:
                self.write_log("\n======= 🔨 Step 1/7: Preprocessing Visium HD h5 to h5ad =======")
                success = self.run_cmd([python_exe, "preprocess.py"])
            elif step == 2:
                self.write_log("\n======= 🌲 Step 2/7: Running scGPT Gene Expression Encoding =======")
                success = self.run_cmd([python_exe, "example.py"], cwd="./scGPT", throttle=True)
            elif step == 3:
                self.write_log("\n======= 📷 Step 3/7: Running UNI Image Feature Extraction =======")
                success = self.run_cmd([python_exe, "uni.py"], cwd="./UNI", throttle=True)
            elif step == 4:
                self.write_log("\n======= 🔬 Step 4/7: Running StarDist Nuclei Mask Prediction =======")
                success = self.run_cmd([python_exe, "run_stardist.py"])
            elif step == 5:
                self.write_log("\n======= 🏷️ Step 5/7: Generating Seeds and Extracting Labeled Boundary Data =======")
                success = self.run_cmd([python_exe, "seed.py"])
                if success:
                    success = self.run_cmd([python_exe, "label.py"])
                if success:
                    success = self.run_cmd([python_exe, "cutlabel.py"])
            elif step == 6:
                self.write_log("\n======= 🧠 Step 6/7: Training Graph Neural Network GNN Joint Model =======")
                success = self.run_cmd([python_exe, "train.py"])
            elif step == 7:
                self.write_log("\n======= 🌊 Step 7/7: Running Cell Diffusion Segmentation via Disjoint-Set and Constrained Distance Algorithms =======")
                success = self.run_cmd([python_exe, "seg.py"])

            if not success:
                self.write_log(f"\n❌ Error: Execution failed at step {step}, pipeline terminated.\n")
                messagebox.showerror("Execution Error", f"Step {step} execution failed. Please check the console logs for details.")
                self.btn_run_all.config(state="normal")
                return

            current_progress += 1
            self.progress["value"] = (current_progress / total_steps) * 100

        self.write_log("\n🎉 Congratulations! All selected computational steps have completed successfully.\n")
        messagebox.showinfo("Success", "The selected cell segmentation and extraction task has completed successfully.")
        self.btn_run_all.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = PipelineGUI(root)
    root.mainloop()
