# AI Trainer

This project provides a set of experiments for analysing human pose with OpenCV, MediaPipe and PyTorch. It compares a reference exercise video against live camera input and gives verbal recommendations.

## Requirements

All required Python packages are listed in `requirements.txt`.

Install them with:

```bash
pip install -r requirements.txt
```

## Usage

The simplest demo can be launched with:

```bash
python ai2.py
```

After starting the program select a reference video (e.g. `source.mp4`) when prompted. The webcam feed will be compared against the reference, recommendations will be displayed on the screen and voiced via `pyttsx3`.

Some scripts in the repository contain additional utilities for preprocessing data (`ai.py`, `main.py`, etc.). Feel free to explore them for experiments.
