✋ Real-Time Sign Language Digit Recognition
📘 Project Overview

This project uses computer vision and deep learning to recognize American Sign Language (ASL) digits (0–9) in real-time using a webcam.
It leverages MediaPipe for hand landmark detection and a TensorFlow neural network for digit classification.

When you show a hand sign (0–9) to your webcam, the system detects your hand, extracts landmarks, classifies the sign, and displays the predicted digit live on screen.

🧰 Tools and Dataset

Tools Used
Python 3.12+
VS Code (for coding and running scripts)
OpenCV (for video capture and image handling)
MediaPipe (for real-time hand tracking)
TensorFlow/Keras (for model training and prediction)

Dataset
https://www.kaggle.com/datasets/rayeed045/american-sign-language-digit-dataset
American Sign Language Digit Dataset
This dataset contains thousands of labeled images for hand signs representing digits 0–9.
Each digit folder contains real images of the corresponding sign.

Python Libraries and Their Roles
Library	Purpose
opencv-python	Captures webcam feed and processes frames
mediapipe	Detects and tracks hand landmarks
tensorflow / keras	Builds, trains, and runs the digit classification model
numpy	Handles numerical and matrix operations
scikit-learn	Used for dataset splitting and evaluation

📁 File Layout
Numbers/
│
├── datasets/
│   └── SignLanguageDigitsDataset/
│       ├── 0/
│       │   └── Input Images - Sign 0/
│       ├── 1/
│       ├── ...
│       └── 9/
│
├── project/
│   ├── data/
│   │   ├── X.npy        # Extracted landmark features
│   │   └── y.npy        # Corresponding labels
│   └── models/
│       └── model.h5     # Saved trained model
│
├── preprocess_digits.py  # Extracts MediaPipe hand landmarks and saves X, y arrays
├── train.py              # Trains and saves CNN model
├── realtime.py           # Live prediction using webcam
└── README.md             # Project documentation

🚀 How to Run the Project (Step-by-Step)

1️⃣ Install Dependencies
Open PowerShell or VS Code Terminal inside your project folder and run:
pip install opencv-python mediapipe tensorflow numpy scikit-learn pyttsx3

2️⃣ Prepare the Dataset
Download the dataset from:
🔗 https://www.kaggle.com/datasets/rayeed045/american-sign-language-digit-dataset
Extract it to:
Numbers/datasets/SignLanguageDigitsDataset/

3️⃣ Run Preprocessing
Extract hand landmark features using MediaPipe:
python preprocess_digits.py

4️⃣ Train the Model
python train.py
This trains a neural network on the extracted features and saves model.h5 to project/models/

5️⃣ Run Real-Time Recognition
python realtime.py
Hold your hand up to your webcam showing digits 0–9 in ASL form.
The script displays the predicted digit live on the video feed.

Press ctrl+cto exit.

💡 Additional Notes
Ensure your lighting is good for accurate hand tracking.
Only one hand should be visible in the frame.
If you get a shape mismatch error, ensure you’re using the updated realtime.py that extracts 42 features (one hand).
You can optionally enable text-to-speech via pyttsx3 in the realtime script.
Works best at 720p resolution and normal lighting conditions.
