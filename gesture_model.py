import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Load dataset (replace with your ASL dataset)
def load_data(dataset_path):
    X, y = [], []
    for label, gesture in enumerate(os.listdir(dataset_path)):
        for img_path in os.listdir(os.path.join(dataset_path, gesture)):
            img = cv2.imread(os.path.join(dataset_path, gesture, img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img)
            if results.multi_hand_landmarks:
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                X.append(landmarks)
                y.append(label)
    return np.array(X), np.array(y)

# Build CNN model
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')  # 10 classes for gestures
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and save model
def train_gesture_model():
    X, y = load_data("path_to_dataset")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    model.save("models/gesture_model.h5")

if __name__ == "__main__":
    train_gesture_model()