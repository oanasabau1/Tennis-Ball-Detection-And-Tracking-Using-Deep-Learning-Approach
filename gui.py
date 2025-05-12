import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import subprocess
import os
import sys
import cv2
from PIL import Image, ImageTk

class TennisAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tennis Match Analyzer")
        self.root.geometry("800x600")
        self.video_path = ""

        self.create_widgets()

    def create_widgets(self):
        # Load video
        self.btn_load = tk.Button(self.root, text="Load Video", command=self.load_video)
        self.btn_load.pack(pady=5)

        self.label_video = tk.Label(self.root, text="No video selected.")
        self.label_video.pack()

        # Run analysis
        self.btn_run = tk.Button(self.root, text="Run Analysis", command=self.run_analysis)
        self.btn_run.pack(pady=5)

        # Output log
        self.output_box = scrolledtext.ScrolledText(self.root, height=10, wrap=tk.WORD)
        self.output_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=False)

        # Video display
        self.video_label = tk.Label(self.root)
        self.video_label.pack(pady=10)

        # Preview buttons
        self.btn_preview_output = tk.Button(self.root, text="Preview Output Video", command=self.preview_output_video)
        self.btn_preview_output.pack(pady=2)

        self.btn_preview_mini = tk.Button(self.root, text="Preview Mini-Court Video", command=self.preview_mini_video)
        self.btn_preview_mini.pack(pady=2)

    def load_video(self):
        path = filedialog.askopenfilename(title="Select a Video File", filetypes=[("MP4 files", "*.mp4")])
        if path:
            self.video_path = path
            self.label_video.config(text=f"Selected: {os.path.basename(path)}")
        else:
            self.label_video.config(text="No video selected.")

    def run_analysis(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Please load a video first.")
            return

        self.output_box.delete(1.0, tk.END)
        self.output_box.insert(tk.END, f"Running analysis on:\n{self.video_path}\n\n")

        try:
            result = subprocess.run(
                [sys.executable, "main.py", self.video_path],
                capture_output=True,
                text=True
            )
            self.output_box.insert(tk.END, result.stdout)
            if result.stderr:
                self.output_box.insert(tk.END, "\nErrors:\n" + result.stderr)
            self.output_box.insert(tk.END, "\n\n✅ Analysis complete.")
        except Exception as e:
            self.output_box.insert(tk.END, f"\n❌ Error during processing:\n{e}")

    def preview_output_video(self):
        self._play_video("output_videos/output_video.avi")

    def preview_mini_video(self):
        self._play_video("output_videos/mini_court_video.avi")

    def _play_video(self, video_path):
        if not os.path.exists(video_path):
            messagebox.showerror("File Not Found", f"{video_path} not found.")
            return

        cap = cv2.VideoCapture(video_path)

        def show_frame():
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 360))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
                self.root.after(40, show_frame)  # ~25 FPS
            else:
                cap.release()

        show_frame()

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = TennisAnalysisApp(root)
    root.mainloop()
