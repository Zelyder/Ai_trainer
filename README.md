# AI Trainer


This repository contains various experiments for processing pose data.

## Running tests

To run the unit tests, install `pytest` and execute:

```bash
pytest
```

This repository contains experiments with pose estimation and
recommendation models for analyzing human motion.

## License

Distributed under the [MIT License](LICENSE).

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

After starting the program select a reference video (e.g. `source.mp4`) when prompted. The webcam feed will be compared against the reference, recommendations will be displayed on the screen and voiced via `pyttsx3`.  Use the **Tolerance** slider in the window to adjust the value passed to `generate_recommendations`.  The slider starts at 15 degrees by default.

Some scripts in the repository contain additional utilities for preprocessing data (`ai.py`, `main.py`, etc.). Feel free to explore them for experiments.

## Simple console analysis

To analyse a prerecorded video without using the webcam run:

```bash
python cli_app.py path/to/video.mp4
```

The script will print the average deviation of four basic joint angles relative
to the first frame and display a graph with the deviation dynamics.



## LSTM Training Example

A simple example of training an LSTM classifier on random data is provided in
`lstm_train.py`. It shows how to track validation accuracy and plot training
curves:

```bash
python lstm_train.py
```

The script writes validation accuracy values to `val_accuracy.txt` and saves
`loss.png` and `accuracy.png` with the corresponding graphs.

## Models

`pose_lstm.py` provides a minimal LSTM classifier implemented in PyTorch. It accepts a sequence of pose features and predicts three categories: correct execution, error A and error B.

## Exercise recommendations

The helper function `generate_recommendations` in `format.py` analyzes several
joint angles between the user's pose and a reference pose. It covers elbows,
knees, shoulders and hips.  Pass a ``tolerance`` value (in degrees) to control
how much deviation is allowed before a textual hint is returned.  When no
tolerance is supplied the default of 10 degrees is used.

## DTW-based pose comparison

`pipeline.py` can compare two videos frame by frame using Dynamic Time Warping
(DTW).  The script extracts the skeleton sequence from each video and prints a
DTW similarity score.  Lower values mean the movements are closer together,
with a perfect match resulting in `0`.  It also plots the joint angle
trajectories of both skeletons so you can visually judge how similar the
motions are.

Run the pipeline from the command line:

```bash
python pipeline.py reference.mp4 session.mp4
```

Or call the function directly from Python:

```python
from pipeline import run_pipeline

run_pipeline("reference.mp4", "session.mp4")
```



