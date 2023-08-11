import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,max_num_hands=2,min_detection_confidence=0.8,min_tracking_confidence=0.5)
hands_videos = mp_hands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.8,min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


def detect_hand_landmarks(image, hands, draw=True, display=True):

    out_img = image.copy()
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks and draw:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image=out_img, landmark_list=hand_landmarks, connections=mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))
    if display:
        plt.figure(figsize=[15,15])
        plt.subplot(121),
