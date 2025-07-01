import os
import cv2
from yt_dlp import YoutubeDL
import sys


def download_youtube_video(url, custom_name, output_dir="downloaded_youtube_videos"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    temp_path = os.path.join(output_dir, '%(title)s.%(ext)s')
    ydl_opts = {
        'outtmpl': temp_path,
        'format': 'bestvideo[ext=mp4]',
        'merge_output_format': 'mp4'
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_file = ydl.prepare_filename(info)
            if custom_name:
                filename = f"{custom_name}.mp4"
                final_path = os.path.join(output_dir, filename)
                os.rename(downloaded_file, final_path)
            else:
                final_path = downloaded_file
            print(f"Downloaded: {final_path}")
            return final_path
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None


def trim_video(input_path, output_dir="input_videos", max_duration_seconds=7, skip_seconds=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    name_without_ext = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{name_without_ext}.mp4")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Failed to open video: {input_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(fps * skip_seconds)
    max_frames = int(fps * max_duration_seconds)
    trim_frames = min(total_frames - start_frame, max_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while frame_count < trim_frames:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached early.")
            break
        out.write(frame)
        frame_count += 1
    cap.release()
    out.release()
    print(f"Trimmed video saved to: {output_path}")
    return output_path


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python process_video_from_youtube.py <YouTube URL> [custom_filename]")
        sys.exit(1)

    url = sys.argv[1]
    custom_name = sys.argv[2] if len(sys.argv) > 2 else None
    downloaded_path = download_youtube_video(url, custom_name=custom_name)
    if downloaded_path:
        trimmed_path = trim_video(downloaded_path, skip_seconds=7)
        if trimmed_path:
            print(f"[TRIMMED_PATH]{trimmed_path}")
