import cv2
import os


video_filename = '../new_video.mp4'
max_duration_seconds = 10


input_path = f'{video_filename}'
name_without_ext = os.path.splitext(os.path.basename(video_filename))[0]
output_folder = '../input_videos'
os.makedirs(output_folder, exist_ok=True)


output_path = os.path.join(output_folder, f'{name_without_ext}.mp4')

cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print(f"Failed to open video: {input_path}")
    exit()


fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
max_frames = int(fps * max_duration_seconds)
trim_frames = min(total_frames, max_frames)

print(f"FPS: {fps}, total input frames: {total_frames}, resolution: {width}x{height}")
print(f"Will process {trim_frames} frames ({trim_frames / fps:.2f} seconds)")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
while frame_count < trim_frames:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached before expected.")
        break

    out.write(frame)
    frame_count += 1

cap.release()
out.release()
print(f"Done! Trimmed video saved to: {output_path}")
