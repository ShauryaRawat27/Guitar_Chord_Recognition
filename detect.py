import cv2
import mediapipe as mp
import pandas as pd
import csv
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open CSV file
csv_filename = "dataset.csv"
fieldnames = ["Thumb_X", "Thumb_Y", "Index_X", "Index_Y", "Middle_X", "Middle_Y", 
              "Ring_X", "Ring_Y", "Pinky_X", "Pinky_Y", "Chord"]
with open(csv_filename, mode='a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    if file.tell() == 0:
        writer.writeheader()  

    print("Press 'c' for C chord, 'g' for G chord, 'd' for D chord... (Press 'q' to quit)")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        key = cv2.waitKey(1) & 0xFF
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                finger_positions = []
                for i in [4, 8, 12, 16, 20]:  # Thumb, Index, Middle, Ring, Pinky
                    finger_positions.append(hand_landmarks.landmark[i].x)
                    finger_positions.append(hand_landmarks.landmark[i].y)

                
                
                chord_label = None
                if key == ord('c'):
                    chord_label = "C"
                elif key == ord('g'):
                    chord_label = "G"
                elif key == ord('d'):
                    chord_label = "D"

                if chord_label:
                    data = dict(zip(fieldnames[:-1], finger_positions))  # Map coordinates
                    data["Chord"] = chord_label
                    writer.writerow(data)
                    print(f"Saved {chord_label} chord data!")

        cv2.imshow("Hand Tracking - Data Collection", frame)
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
