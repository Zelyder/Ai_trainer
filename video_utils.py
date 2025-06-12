import cv2
import os


def save_video_frames(video_path, output_dir, step=1):
    """Save frames from a video to a directory every ``step`` frames.

    Parameters
    ----------
    video_path : str
        Path to the source video file.
    output_dir : str
        Directory where extracted frames will be saved. Created if missing.
    step : int, optional
        Interval between saved frames. ``1`` saves every frame.

    Returns
    -------
    int
        Number of frames saved to ``output_dir``.
    """
    if step < 1:
        raise ValueError("step must be >= 1")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            filename = os.path.join(output_dir, f"frame_{idx:06d}.png")
            cv2.imwrite(filename, frame)
            saved += 1
        idx += 1

    cap.release()
    return saved
