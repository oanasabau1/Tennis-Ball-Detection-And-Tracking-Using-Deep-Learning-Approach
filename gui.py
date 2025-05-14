import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox, scrolledtext
import subprocess
import os
import sys
import cv2
import threading
from PIL import Image, ImageTk
import shutil
import time


class TennisAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tennis Match Analyzer")
        self.root.attributes('-fullscreen', True)
        self.video_name = ""
        self.video_path = ""

        self.create_start_menu()

    def clear_root(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def create_start_menu(self):
        self.clear_root()

        style = ttk.Style()
        style.configure("TFrame", padding=10)
        style.configure("Title.TLabel", font=("Verdana", 34, "bold"))

        main_frame = ttk.Frame(self.root, style="TFrame")
        main_frame.pack(fill="both", expand=True)

        exit_button = ttk.Button(
            main_frame,
            text="✕",
            bootstyle="danger",
            command=self.root.quit,
            width=3
        )
        exit_button.place(relx=0.98, rely=0.02, anchor="ne")

        content_frame = ttk.Frame(main_frame)
        content_frame.place(relx=0.5, rely=0.45, anchor="center")

        image_frame = ttk.Frame(content_frame)
        image_frame.pack(pady=30)
        self.logo_images = []
        image_files = ["tennis_racket.png", "tennis_court.jpg", "girl_player.jpg"]
        for image_file in image_files:
            image_path = os.path.join("assets", image_file)
            try:
                image_outer = ttk.Frame(image_frame, bootstyle="default")
                image_outer.pack(side="left", padx=20)

                image_container = ttk.Frame(
                    image_outer,
                    bootstyle="default",
                    padding=3
                )
                image_container.pack(padx=2, pady=2)
                img = Image.open(image_path)
                img = img.resize((220, 220), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.logo_images.append(photo)
                image_label = ttk.Label(
                    image_container,
                    image=photo,
                )
                image_label.pack()
            except Exception as e:
                print(f"Could not load image {image_file}: {e}")

        title = ttk.Label(
            content_frame,
            text="Tennis Match Analyzer",
            style="Title.TLabel"
        )
        title.pack(pady=25)

        button_frame = ttk.Frame(content_frame)
        button_frame.pack(pady=30)
        start_button = ttk.Button(
            button_frame,
            text="Start",
            bootstyle="success-outline",
            command=self.create_main_ui,
            width=25
        )
        start_button.pack(pady=12, ipady=12)

    def create_main_ui(self):
        self.clear_root()

        # Configure styles
        style = ttk.Style()
        style.configure("TLabel", font=("Verdana", 12))

        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Top controls section
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill="x", pady=10)

        # Left side controls
        controls_frame = ttk.Frame(top_frame)
        controls_frame.pack(side="left", padx=(0, 20))

        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(fill="x")

        ttk.Button(button_frame, text="Load Video", command=self.load_video,
                   bootstyle="primary-outline", width=25).pack(pady=5)
        ttk.Button(button_frame, text="Download From YouTube", command=self.open_youtube_window,
                   bootstyle="info-outline", width=25).pack(pady=5)
        self.btn_run = ttk.Button(button_frame, text="Run Analysis", command=self.run_analysis,
                                  bootstyle="success-outline", width=25)
        self.btn_run.pack(pady=5)

        # Right side status info
        info_frame = ttk.Frame(top_frame)
        info_frame.pack(side="left", fill="x", expand=True)

        self.label_video = ttk.Label(info_frame, text="No video selected.")
        self.label_video.pack(anchor="w", pady=5)
        self.status_label = ttk.Label(info_frame, text="")
        self.status_label.pack(anchor="w", pady=5)

        # Middle section - Videos
        video_container = ttk.LabelFrame(main_frame, text="Video Analysis", bootstyle="default")
        video_container.pack(fill="both", expand=True, pady=10)

        # Center the videos with a flexible layout
        video_frame = ttk.Frame(video_container)
        video_frame.pack(fill="both", expand=True)
        video_frame.columnconfigure(0, weight=1)
        video_frame.columnconfigure(1, weight=1)
        video_frame.rowconfigure(0, weight=1)

        # Output video (left)
        output_frame = ttk.Frame(video_frame)
        output_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=0)
        output_frame.rowconfigure(1, weight=1)

        ttk.Label(output_frame, text="Output Video").grid(row=0, column=0, sticky="w")
        self.output_label = ttk.Label(output_frame, borderwidth=2, relief="groove")
        self.output_label.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Court analysis (right)
        mini_frame = ttk.Frame(video_frame)
        mini_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        mini_frame.columnconfigure(0, weight=1)
        mini_frame.rowconfigure(0, weight=0)
        mini_frame.rowconfigure(1, weight=1)

        ttk.Label(mini_frame, text="Court Analysis").grid(row=0, column=0, sticky="w")
        self.mini_label = ttk.Label(mini_frame, borderwidth=2, relief="groove")
        self.mini_label.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Bottom section - Analysis results
        speed_frame = ttk.LabelFrame(main_frame, text="Shot Analysis", bootstyle="primary")
        speed_frame.pack(fill="x", pady=10)

        self.speed_box = scrolledtext.ScrolledText(
            speed_frame, height=8, wrap='word',
            font=("Consolas", 12), background="#f8f9fa"
        )
        self.speed_box.pack(padx=10, pady=10, fill='both')

        # Set up event handlers
        self.root.bind("<Escape>",
                       lambda e: self.root.protocol("WM_RESIZE_WINDOW", self.root.attributes('-fullscreen', False)))
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.youtube_url_var = ttk.StringVar()

    def open_youtube_window(self):
        win = ttk.Toplevel(self.root)
        win.title("Download Video from YouTube")
        win.geometry("400x200")

        ttk.Label(win, text="YouTube URL:").pack(pady=5)
        url_entry = ttk.Entry(win, width=50)
        url_entry.pack(pady=5)

        ttk.Label(win, text="Custom Filename:").pack(pady=5)
        name_entry = ttk.Entry(win, width=50)
        name_entry.pack(pady=5)

        def start_download():
            url = url_entry.get()
            name = name_entry.get()
            if not url or not name:
                messagebox.showwarning("Missing Input", "Please enter both URL and filename.")
                return
            self.status_label.config(text="Downloading video...")
            self.root.update_idletasks()
            thread = threading.Thread(target=self._download_youtube_only, args=(url, name))
            thread.start()
            win.destroy()

        ttk.Button(win, text="Download", command=start_download, bootstyle=SUCCESS).pack(pady=10)

    def _download_youtube_only(self, url, name):
        try:
            result = subprocess.run([
                sys.executable,
                os.path.join("process_video", "process_video_from_youtube.py"),
                url,
                name
            ], capture_output=True, text=True)

            self.speed_box.insert('end', result.stdout)
            if result.stderr:
                self.speed_box.insert('end', f"\nErrors:\n{result.stderr}")

            if result.returncode != 0:
                self.status_label.config(text="Error during download.", foreground="red")
                return

            trimmed_path = None
            for line in result.stdout.splitlines():
                if line.startswith("[TRIMMED_PATH]"):
                    trimmed_path = line.replace("[TRIMMED_PATH]", "").strip()
                    break

            if not trimmed_path or not os.path.exists(trimmed_path):
                self.status_label.config(text="Trimmed video not found.", foreground="red")
                return

            self.video_name = os.path.splitext(os.path.basename(trimmed_path))[0]
            self.status_label.config(text="Video downloaded and trimmed.", foreground="green")
            self.label_video.config(text=f"Downloaded: {self.video_name}")

        except Exception as e:
            self.speed_box.insert('end', f"\nException: {e}")
            self.status_label.config(text="Error occurred.", foreground="red")

    def load_video(self):
        path = filedialog.askopenfilename(title="Select a Video File", filetypes=[("MP4 files", "*.mp4")])
        if not path:
            return

        if not os.path.exists(path):
            messagebox.showerror("File Error", "The selected file does not exist.")
            return

        self.video_path = path
        self.video_name = os.path.splitext(os.path.basename(path))[0]

        os.makedirs("input_videos", exist_ok=True)
        dest = os.path.join("input_videos", f"{self.video_name}.mp4")

        if os.path.abspath(path) == os.path.abspath(dest):
            self.label_video.config(text=f"Selected: {self.video_name}")
            return

        try:
            with open(path, 'rb'):
                pass
            if not self.try_copy_file(path, dest):
                raise PermissionError("Could not copy file after several attempts.")
        except Exception as e:
            messagebox.showerror("File Error", f"Error copying file:\n{e}")
            self.label_video.config(text="Copy failed.")
            return

        self.label_video.config(text=f"Selected: {self.video_name}")

    def try_copy_file(self, src, dst, retries=3, delay=1):
        for _ in range(retries):
            try:
                shutil.copy2(src, dst)
                return True
            except PermissionError:
                time.sleep(delay)
        return False

    def run_analysis(self):
        if not self.video_name:
            messagebox.showwarning("No Video", "Please load or download a video first.")
            return

        self.speed_box.delete(1.0, 'end')
        self.speed_box.insert('end', f"Running analysis_of_tennis_ball on: {self.video_name}\n\n")
        self.status_label.config(text="Processing... Please wait ⏳")
        self.btn_run.config(state='disabled')

        thread = threading.Thread(target=self._process_video)
        thread.start()

    def _process_video(self):
        try:
            self.speed_box.delete(1.0, 'end')

            result = subprocess.run([
                sys.executable, "main.py", self.video_name
            ], capture_output=True, text=True)

            shot_info = []

            for line in result.stdout.splitlines():
                if "Shot" in line and "Speed" in line:
                    shot_info.append(line)
                elif "=== Shot Stats ===" in line:
                    shot_info.append("\n" + line)
                elif "Number of shots:" in line or "Average speed:" in line:
                    shot_info.append(line)

            self.speed_box.insert('end', "\n".join(shot_info))

            if result.stderr:
                self.status_label.config(text="Warnings during analysis_of_tennis_ball.", foreground="orange")

            if result.returncode != 0:
                self.status_label.config(text="Analysis failed.", foreground="red")
                return

            self.status_label.config(text="Analysis complete.", foreground="green")
            self.play_both_videos()
        except Exception as e:
            self.speed_box.insert('end', f"\nError: {e}")
            self.status_label.config(text="Error occurred.", foreground="red")
        finally:
            self.btn_run.config(state='normal')

    def play_both_videos(self):
        self.release_resources()

        output_path = f"outputs/{self.video_name}/{self.video_name}.avi"
        mini_path = f"outputs/{self.video_name}/mini_court_for_{self.video_name}.avi"

        if not os.path.exists(output_path) or not os.path.exists(mini_path):
            self.speed_box.insert('end', "Processed videos not found.\n")
            return

        self.output_cap = cv2.VideoCapture(output_path)
        self.mini_cap = cv2.VideoCapture(mini_path)
        self.show_frames()

    def show_frames(self):
        if not hasattr(self, 'output_cap') or not hasattr(self, 'mini_cap'):
            return

        ret1, frame1 = self.output_cap.read()
        ret2, frame2 = self.mini_cap.read()

        if ret1:
            frame1 = cv2.resize(frame1, (640, 360))
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            img1 = ImageTk.PhotoImage(Image.fromarray(frame1))
            self.output_label.imgtk = img1
            self.output_label.config(image=img1)

        if ret2:
            frame2 = cv2.resize(frame2, (640, 360))
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            img2 = ImageTk.PhotoImage(Image.fromarray(frame2))
            self.mini_label.imgtk = img2
            self.mini_label.config(image=img2)

        if ret1 or ret2:
            self.root.after(40, self.show_frames)
        else:
            self.output_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.mini_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.root.after(40, self.show_frames)

    def release_resources(self):
        if hasattr(self, 'output_cap'):
            self.output_cap.release()
        if hasattr(self, 'mini_cap'):
            self.mini_cap.release()

    def on_closing(self):
        self.release_resources()
        self.root.destroy()


if __name__ == "__main__":
    root = ttk.Window(themename="flatly")
    app = TennisAnalysisApp(root)
    root.mainloop()