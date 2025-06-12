import cv2
import json
import mediapipe as mp


def save_blazepose_keypoints(video_path, output_json, max_frames=None):
    """Extract 33 BlazePose landmarks for each frame and store them as JSON.

    Parameters
    ----------
    video_path : str or int
        Path to the input video file or webcam index for ``cv2.VideoCapture``.
    output_json : str
        Path where the resulting JSON will be written.
    max_frames : int, optional
        If given, only the first ``max_frames`` frames will be processed.
    """
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    results = []

    with mp_pose.Pose(static_image_mode=False) as pose:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmarks = []
            if processed.pose_landmarks:
                for lm in processed.pose_landmarks.landmark:
                    landmarks.append({
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility,
                    })
            else:
                landmarks = [{"x": None, "y": None, "z": None, "visibility": 0.0} for _ in range(33)]

            results.append(landmarks)
            frame_idx += 1
            if max_frames is not None and frame_idx >= max_frames:
                break

    cap.release()

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

