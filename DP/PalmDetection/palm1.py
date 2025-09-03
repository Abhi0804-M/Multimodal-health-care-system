import numpy as np
import mediapipe as mp
import cv2
import os
import time

# Create a directory to store the saved images
if not os.path.exists('captured_photos'):
    os.makedirs('captured_photos')

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

start_time = None
capture_time = 3  # 3 seconds for capturing the image

success = True
while success:
    success, img = cap.read()
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Select specific landmarks: 5, 9, 13, 17, 0
            landmarks_of_interest = [5, 9, 13, 17, 0]

            # Check hand openness using WRIST and INDEX_FINGER_TIP landmarks
            wrist_landmark = hand_landmarks.landmark[mpHands.HandLandmark.WRIST]
            tip_landmark = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate the Euclidean distance between wrist and tip landmarks
            distance = np.sqrt((wrist_landmark.x - tip_landmark.x) ** 2 + (wrist_landmark.y - tip_landmark.y) ** 2)

            # If the distance is below a threshold, consider the hand closed
            if distance < 0.03:  # You may need to adjust this threshold based on your observations
                continue  # Skip processing if the hand is closed

            # Extract y coordinates for selected landmarks
            y_coordinates = [hand_landmarks.landmark[idx].y * img.shape[0] for idx in landmarks_of_interest]

            # Identify the palmar side dynamically
            if y_coordinates[0] < y_coordinates[1] < y_coordinates[2] < y_coordinates[3] < y_coordinates[4]:
                palmar_side = True
            else:
                palmar_side = False

            # Check if the dorsal side is facing up
            if not palmar_side:
                # Calculate the convex hull of the specified landmarks
                hull = cv2.convexHull(np.array([(int(hand_landmarks.landmark[idx].x * img.shape[1]), int(hand_landmarks.landmark[idx].y * img.shape[0])) for idx in landmarks_of_interest]))

                # Draw the convex hull (frame) around the specified landmarks
                cv2.drawContours(img, [hull], -1, (0, 255, 0), 2)

                # Draw connections between selected landmarks
                connections = [(5, 9), (9, 13), (13, 17), (17, 0), (0, 5)]
                for connection in connections:
                    x1, y1 = int(hand_landmarks.landmark[connection[0]].x * img.shape[1]), int(
                        hand_landmarks.landmark[connection[0]].y * img.shape[0])
                    x2, y2 = int(hand_landmarks.landmark[connection[1]].x * img.shape[1]), int(
                        hand_landmarks.landmark[connection[1]].y * img.shape[0])
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Check if it's the palmar side and 3 seconds have passed
            if not palmar_side and start_time is not None and time.time() - start_time >= capture_time:
                # Create a mask for the region enclosed by the landmarks
                mask = np.zeros_like(img)
                cv2.fillPoly(mask, [np.array([(int(hand_landmarks.landmark[idx].x * img.shape[1]), int(hand_landmarks.landmark[idx].y * img.shape[0])) for idx in landmarks_of_interest])], (255, 255, 255))

                # Apply the mask to the original image to get the palm region
                palm_region = cv2.bitwise_and(img, mask)

                # Save the palm region as a PNG file
                cv2.imwrite('captured_photos/palm_region.png', palm_region)
                start_time = None  # Reset the start time
                success = False  # Exit the loop after capturing the image

            if not palmar_side:
                # Start the timer if it's the palmar side
                if start_time is None and not palmar_side:
                    start_time = time.time()
                

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
