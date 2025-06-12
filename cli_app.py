import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

from ai2 import calc_angles_ntu, extract_keypoints


def analyse_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open {video_path}")

    angles_seq = []
    with mp.solutions.pose.Pose(static_image_mode=False) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pts = extract_keypoints(frame, pose)
            if pts is None or len(pts) < 17:
                continue
            angles = calc_angles_ntu(pts)
            angles_seq.append(angles)
    cap.release()

    if not angles_seq:
        raise RuntimeError("No poses detected in the video")

    reference = angles_seq[0]
    deviations = [np.abs(a - reference) for a in angles_seq]
    deviations = np.array(deviations)
    mean_deviation = deviations.mean(axis=0)

    return deviations, mean_deviation


def plot_deviations(deviations):
    plt.figure(figsize=(10, 6))
    for i in range(deviations.shape[1]):
        plt.plot(deviations[:, i], label=f'Joint {i+1}')
    plt.xlabel('Frame')
    plt.ylabel('Deviation (degrees)')
    plt.title('Angle Deviations from First Frame')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyse a video and show joint deviations.')
    parser.add_argument('video', help='Path to video file')
    args = parser.parse_args()

    deviations, mean_dev = analyse_video(args.video)
    for i, dev in enumerate(mean_dev):
        print(f'Average deviation for joint {i+1}: {dev:.1f} degrees')
    plot_deviations(deviations)


if __name__ == '__main__':
    main()
