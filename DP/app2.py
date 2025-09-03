from flask import Flask, request, render_template, redirect, url_for, jsonify 
from roboflow import Roboflow
from werkzeug.utils import secure_filename
import supervision as sv
from sklearn.preprocessing import MinMaxScaler
import cv2
import numpy as np
import keras 
from keras._tf_keras.keras.models import load_model
import os
from PIL import Image
import threading
import mediapipe as mp
import time
import re
from matplotlib import pyplot as plt
app = Flask(__name__,template_folder='D:/DP/templates')
app.config['UPLOAD_FOLDER'] = 'D:/DP/uploads'
k=0
# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the anemia Classification models

iron_deficiency_model = load_model('iron_deficiency_model.h5')
vitamin_deficiency_model = load_model('vitamin_deficiency_model.h5')
sickle_cell_model = load_model('sickle_cell_model.h5')
anemia_model= load_model('anemia_model.h5')

scaler = MinMaxScaler()
scaler.fit([[80, 27, 11], [100, 30, 11]])
# Load the anemia prediction models
anemia_eye_model = load_model('conjunctiva_100.h5')
anemia_nail_model = load_model('fingerNail_100.h5')
anemia_palm_model = load_model('palm_100.h5')

def is_mobile():
    user_agent = request.headers.get('User-Agent', '').lower()
    mobile_keywords = ['android', 'iphone', 'ipad', 'ipod', 'blackberry', 'windows phone', 'opera mini', 'mobile']
    return any(keyword in user_agent for keyword in mobile_keywords)

@app.route('/')
def index():
    global k
    if is_mobile():
        
        k=0
        print(k) # Serve a mobile-friendly version
    else:
        
        k=1
        print(k)
    return render_template('index.html')  # Example placeholder route

@app.route('/anemic', methods=['GET'])
def anemic():
    return render_template('anemia.html')

@app.route('/Classify', methods=['GET','POST'])
def Classify():
    return render_template('Classify.html')

@app.route('/Classify1', methods=['POST'])
def Classify1():
    data = request.get_json()
    mcv = float(data['mcv'])
    mch = float(data['mch'])
    hgb = float(data['hgb'])

    # Convert the input values to a numpy array and reshape into a single sample with 3 features
    new_input_values = np.array([mcv, mch, hgb]).reshape(1, -1)

    # Scale the new input features using the MinMaxScaler
    new_input_scaled = scaler.transform(new_input_values)

    # Reshape the scaled input data to 3D tensor format for RNN
    new_input_rnn = new_input_scaled.reshape((new_input_scaled.shape[0], 1, new_input_scaled.shape[1]))

    # Make predictions for iron deficiency anemia, vitamin deficiency anemia, and sickle cell anemia
    anemia_probability = anemia_model.predict(new_input_rnn)[0][0]
    iron_deficiency_probability = iron_deficiency_model.predict(new_input_rnn)[0][0]
    vitamin_deficiency_probability = vitamin_deficiency_model.predict(new_input_rnn)[0][0]
    sickle_cell_probability = sickle_cell_model.predict(new_input_rnn)[0][0]

    # Combine probabilities into a single array
    probabilities = np.array([iron_deficiency_probability, vitamin_deficiency_probability, sickle_cell_probability])

    # Normalize the probabilities so they sum to 1
    normalized_probabilities = probabilities / probabilities.sum()

    # Determine the type of anemia with the highest probability
    anemia_types = ['Iron Deficiency Anemia', 'Vitamin Deficiency Anemia', 'Sickle Cell Anemia']
    anemia_info = [
        "Iron deficiency anemia occurs when your body doesn't have enough iron to produce adequate amounts of hemoglobin.",
        "Vitamin deficiency anemia occurs when there are lower-than-normal amounts of vitamins such as B12 or folate.",
        "Sickle cell anemia is a genetic disorder that causes red blood cells to become misshapen and break down."
    ]
    highest_probability_index = np.argmax(normalized_probabilities)
    highest_probability_anemia = anemia_types[highest_probability_index]
    highest_probability_value = normalized_probabilities[highest_probability_index]
    anemia_description = anemia_info[highest_probability_index]
    # Prepare the result to send back to the HTML
    if anemia_probability > 0.5:
        result = {
            'type': highest_probability_anemia,
            'probability': float(highest_probability_value),
            'description': anemia_description
        }

        
    else:
        result = {
            'type': 'Non Anemic',
            'probability': float(0),
            'description': "You are fine"
        }

    print(result)
    return jsonify(result)

@app.route('/nonanemic', methods=['GET'])
def nonanemic():
    return render_template('non_anemia.html')

@app.route('/About', methods=['GET'])
def About():
    return render_template('About.html')

@app.route('/Work', methods=['GET'])
def Work():
    return render_template('Work.html')

@app.route('/contact')
def contact():
    if request.method == 'POST':
        # do stuff when the form is submitted

        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('index'))

    # show the form, it wasn't submitted
    return render_template('contact.html')

@app.route('/eyes')
def eyes():
    result= predict_anemia(anemia_eye_model, r'D:/DP/PalmDetection/captured_photos/palm_region.png', image_size=(128, 128))
    print(result)

def is_mobile():
    user_agent = request.headers.get('User-Agent', '').lower()
    mobile_regex = re.compile(r"android|iphone|ipad|ipod|mobile", re.IGNORECASE)
    return bool(mobile_regex.search(user_agent))

@app.route('/start_capture', methods=['GET'])
def start_capture():
    if not os.path.exists('captured_photos'):
        os.makedirs('captured_photos')
    cap = None 
    if k==0:
# Replace the URL with the video stream link
        url = "http://192.168.29.100:6677/videofeed?username=&password="

        # Open the video stream
        cap = cv2.VideoCapture(url)
    elif k==1:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream")
        return
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    start_time = None
    capture_time = 3  # 3 seconds for capturing the image
    min_brightness = 80  # Minimum acceptable brightness level
    max_brightness = 180  
    min_distance =  650# Minimum pixel distance for hand size
    max_distance = 700
    success = True
    while success:
        success, img = cap.read()
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray_img)

        if avg_brightness < min_brightness or avg_brightness > max_brightness:
            message = f"Adjust lighting! Brightness is {avg_brightness:.2f}. It should be between {min_brightness} and {max_brightness}."
            cv2.putText(img, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[mpHands.HandLandmark.WRIST]
                middle_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP]

                wrist_coords = (int(wrist.x * img.shape[1]), int(wrist.y * img.shape[0]))
                middle_coords = (int(middle_finger_tip.x * img.shape[1]), int(middle_finger_tip.y * img.shape[0]))

                # Euclidean distance between WRIST and MIDDLE_FINGER_TIP
                distance = np.sqrt((wrist_coords[0] - middle_coords[0])**2 + (wrist_coords[1] - middle_coords[1])**2)

                # Check if distance is within acceptable range
                if not (min_distance <= distance <= max_distance):
                    message = f"Adjust distance! Current: {int(distance)}px. It should be between {min_distance}-{max_distance}px."
                    cv2.putText(img, message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('Image', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Select specific landmarks: 5, 9, 13, 17, 0
                landmarks_of_interest = [5, 9, 13, 17, 0, 1, 2, 5]

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
                    connections = [(5, 9), (9, 13), (13, 17), (17, 0), (0, 1), (1, 2), (2, 5)]
                    for connection in connections:
                        x1, y1 = int(hand_landmarks.landmark[connection[0]].x * img.shape[1]), int(hand_landmarks.landmark[connection[0]].y * img.shape[0])
                        x2, y2 = int(hand_landmarks.landmark[connection[1]].x * img.shape[1]), int(hand_landmarks.landmark[connection[1]].y * img.shape[0])
                        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Check if it's the palmar side and 3 seconds have passed
                if not palmar_side and start_time is not None and time.time() - start_time >= capture_time:
                    # Create a mask for the region enclosed by the landmarks
                    mask = np.zeros_like(img)
                    cv2.fillPoly(mask, [np.array([(int(hand_landmarks.landmark[idx].x * img.shape[1]), int(hand_landmarks.landmark[idx].y * img.shape[0])) for idx in landmarks_of_interest])], (255, 255, 255))

                    # Apply the mask to the original image to get the palm region
                    palm_region = cv2.bitwise_and(img, mask)

                    # Save the palm region as a PNG file
                    cv2.imwrite('D:/DP/PalmDetection/captured_photos/palm_region.png', palm_region)
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
        # Call the predict_anemia function when the OK button is clicked
    image_path = "D:/DP/PalmDetection/captured_photos/palm_region.png"
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a binary mask where non-black areas are white
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours of the non-black region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour (assumed to be the palm region)
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a mask of the largest contour (polygon shape)
        polygon_mask = np.zeros_like(gray)
        cv2.drawContours(polygon_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Apply the mask to extract only the palm region
        polygon_cutout = cv2.bitwise_and(image, image, mask=polygon_mask)

        # Get bounding box around the polygon (removes unnecessary black areas)
        x, y, w, h = cv2.boundingRect(largest_contour)
        polygon_cutout = polygon_cutout[y:y+h, x:x+w]  # Crop the image to the bounding box

        # Convert to RGBA to allow transparency
        polygon_cutout_rgba = cv2.cvtColor(polygon_cutout, cv2.COLOR_BGR2BGRA)

        # Set all black pixels (0,0,0) to transparent
        black_pixels = (polygon_cutout_rgba[:, :, 0] == 0) & \
                    (polygon_cutout_rgba[:, :, 1] == 0) & \
                    (polygon_cutout_rgba[:, :, 2] == 0)
          # Make black pixels transparent
        green_pixels = (polygon_cutout_rgba[:, :, 0] == 0) & \
                   (polygon_cutout_rgba[:, :, 1] == 255) & \
                   (polygon_cutout_rgba[:, :, 2] == 0)
        polygon_cutout_rgba[black_pixels | green_pixels, 3] = 0
        # Save the final transparent cutout
        polygon_cutout_path = "captured_photos/palm_transparent.png"
        cv2.imwrite(polygon_cutout_path, polygon_cutout_rgba)
        # Display the final polygonal cutout
        plt.imshow(cv2.cvtColor(polygon_cutout_rgba, cv2.COLOR_BGRA2RGBA))
        plt.axis("off")
        plt.show()

        print(f"Final cropped polygon-segmented palm saved at: {polygon_cutout_path}")
    else:
        print("No valid palm region found!")
    
    result1= predict_anemia(anemia_palm_model, r'captured_photos/palm_transparent.png', image_size=(128, 128))
             # Or handle the result as needed
    print(result1)
    if result1=="Anemic":
        return redirect(url_for('anemic'))
    else:
        return redirect(url_for('nonanemic'))


def predict_anemia(model, img_path, image_size=(128, 128)):
    img = cv2.imread(img_path)
    # cv2.imwrite('D:/DP/PalmDetection/captured_photos/palm_region1.png', palm_region)
    if img is not None:
# Display the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create a binary mask where non-black areas are white
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Find contours of the non-black region
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the largest contour (assumed to be the palm region)
            largest_contour = max(contours, key=cv2.contourArea)

            # Create a mask of the largest contour (polygon shape)
            polygon_mask = np.zeros_like(gray)
            cv2.drawContours(polygon_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

            # Apply the mask to extract only the palm region
            polygon_cutout = cv2.bitwise_and(img, img, mask=polygon_mask)

            # Get bounding box around the polygon (removes unnecessary black areas)
            x, y, w, h = cv2.boundingRect(largest_contour)
            polygon_cutout = polygon_cutout[y:y+h, x:x+w]  # Crop the image to the bounding box

            # Convert to RGBA to allow transparency
            polygon_cutout_rgba = cv2.cvtColor(polygon_cutout, cv2.COLOR_BGR2BGRA)

            # Set all black pixels (0,0,0) to transparent
            black_pixels = (polygon_cutout_rgba[:, :, 0] == 0) & \
                        (polygon_cutout_rgba[:, :, 1] == 0) & \
                        (polygon_cutout_rgba[:, :, 2] == 0)
            # Make black pixels transparent
            green_pixels = (polygon_cutout_rgba[:, :, 0] == 0) & \
                    (polygon_cutout_rgba[:, :, 1] == 255) & \
                    (polygon_cutout_rgba[:, :, 2] == 0)
            polygon_cutout_rgba[black_pixels | green_pixels, 3] = 0
            # Save the final transparent cutout
            polygon_cutout_path = "captured_photos/palm_transparent1.png"
            cv2.imwrite(polygon_cutout_path, polygon_cutout_rgba)
            plt.imshow(cv2.cvtColor(polygon_cutout_rgba, cv2.COLOR_BGRA2RGBA))
            plt.axis("off")
            plt.show()
        image_path="captured_photos/palm_transparent1.png"
        
        img1 = cv2.imread(image_path)
        img = cv2.resize(img1, image_size)
        img = img / 255.0  # Normalize to [0, 1]
        # cv2.imshow("Palm Image", img)
        img_array = np.expand_dims(img, axis=0)  # Add batch dimension

            # Make a prediction
        prediction = model.predict(img_array)
        print(prediction)
            # Convert the prediction to a binary class label
        threshold = 0.5
        class_label = (prediction > threshold).astype(int)
        print(class_label)
            # Output the result
        if class_label == 0:
            return "Not Anemic"
            
        else:
            return "Anemic"
            
    else:
        return "Image could not be loaded"

@app.route('/camera')
def camera():
    return render_template('camera.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if an image file was uploaded
        if 'image' not in request.files:
            return 'No file part'
        file = request.files['image']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            rf = Roboflow(api_key="tnmXe3ZHhmDuk99BH57W")
            project = rf.workspace().project("nakhoon")
            model = project.version(1).model

            try:
                im = Image.open(filepath)
                im = im.convert("RGB")
                im.save(os.path.join(app.config['UPLOAD_FOLDER'], 'nails', 'nail.png'), "PNG", quality=95)

                result = model.predict(os.path.join(app.config['UPLOAD_FOLDER'], 'nails', 'nail.png'), confidence=40).json()
                print(result)  # Debug: Print the result to inspect its structure

                detections = sv.Detections.from_roboflow(result)
                if not detections:
                    print("error happended 1")
                    return render_template('upload.html', error="No nails detected. Please upload a valid image.")
                # Read the image using OpenCV
                image = cv2.imread(filepath)
                if image is None:
                    return "Error: Image could not be loaded"

                bounding_box_annotator = sv.BoundingBoxAnnotator()
                annotated_frame = bounding_box_annotator.annotate(
                    scene=image.copy(),
                    detections=detections
                )

                label_annotator = sv.LabelAnnotator()
                annotated_image = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)

                i = 0
                sv.plot_image(image=annotated_image, size=(16, 16))
                for detection in detections:
                    print(detection)  # Debug: Print each detection to inspect its structure

                    coordinates = detection[0]
                    x, y, w, h = map(int, coordinates)
                    roi = annotated_frame[y:h, x:w]
                    sv.plot_image(image=roi, size=(16, 16))

                    # Save or process the ROI as needed
                    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], f'nails/nail{i}.png'), roi)
                    i += 1
                i= i//2
                result2 = predict_anemia(anemia_nail_model, os.path.join(app.config['UPLOAD_FOLDER'], 'nails', f'nail{i}.png'), image_size=(128, 128))
                print(result2)
                if result2 == "Anemic":
                    return redirect(url_for('anemic'))
                else:
                    return redirect(url_for('nonanemic'))

            except Exception as e:
                return str(e)
    
    return render_template('upload.html')

@app.route('/upload1', methods=['GET', 'POST'])
def upload1():
    if request.method == 'POST':
        # Check if an image file was uploaded
        if 'image' not in request.files:
            return 'No file part'
        file = request.files['image']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            rf = Roboflow(api_key="LNTF1Mjfh8X0v33nx2Tg")
            project = rf.workspace().project("conjunctiva-segmentation-2")
            model = project.version(2).model

            try:
                im = Image.open(filepath)
                im = im.convert("RGB")
                im.save(os.path.join(app.config['UPLOAD_FOLDER'], 'eyes', 'eye.png'), "PNG", quality=95)

                result = model.predict(os.path.join(app.config['UPLOAD_FOLDER'], 'eyes', 'eye.png'), confidence=40).json()
                print(result)  # Debug: Print the result to inspect its structure

                if not result["predictions"]:
                    print("Error: No conjunctiva detected")
                    return render_template('upload1.html', error="No conjunctiva detected. Please upload a valid image.")

                # Read the image using OpenCV
                image = cv2.imread(filepath)
                if image is None:
                    return "Error: Image could not be loaded"

                # ✅ Extract polygon points properly
                polygons = []
                for prediction in result["predictions"]:
                    if "points" in prediction:
                        points = np.array([(point["x"], point["y"]) for point in prediction["points"]], np.int32)
                        polygons.append(points)

                # ✅ If no polygon annotations, return an error
                if not polygons:
                    print("Error: No valid polygon detected")
                    return render_template('upload1.html', error="No conjunctiva detected. Please upload a valid image.")

                # ✅ Annotate the image with polygon detection
                annotated_image = image.copy()
                for polygon in polygons:
                    polygon = polygon.reshape((-1, 1, 2))  # Ensure correct shape
                    cv2.polylines(annotated_image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

                # Save the annotated image
                cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_image.png'), annotated_image)

                # ✅ Extract and Save Conjunctiva Region for Analysis
                i = 0
                for polygon in polygons:
                    # Create a mask for the polygon
                    mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
                    cv2.fillPoly(mask, [polygon], 255)

                    # Extract the conjunctiva region
                    roi = cv2.bitwise_and(image, image, mask=mask)

                    # Crop to bounding box around the polygon
                    x, y, w, h = cv2.boundingRect(polygon)
                    roi_cropped = roi[y:y+h, x:x+w]

                    # Save the extracted conjunctiva
                    roi_path = os.path.join(app.config['UPLOAD_FOLDER'], f'eyes/eye{i}.png')
                    cv2.imwrite(roi_path, roi_cropped)
                    sv.plot_image(image=roi_cropped, size=(16, 16))
                    i += 1

                # ✅ Predict Anemia from Extracted Region
                i = i // 2
                result3 = predict_anemia(anemia_eye_model, os.path.join(app.config['UPLOAD_FOLDER'], 'eyes', f'eye{i}.png'), image_size=(128, 128))
                print(result3)

                # ✅ Redirect Based on Anemia Prediction
                if result3 == "Anemic":
                    return redirect(url_for('anemic'))
                else:
                    return redirect(url_for('nonanemic'))

            except Exception as e:
                return str(e)

    return render_template('upload1.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
