from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np

# Initialize Roboflow and load the model
rf = Roboflow(api_key="LNTF1Mjfh8X0v33nx2Tg")
project = rf.workspace().project("conjunctiva-segmentation-2")
model = project.version(2).model

# Predict using the model
result = model.predict("D:/Proj/Conjuctiva/istockphoto-1388917616-612x612.jpg", confidence=40).json()

# Extract labels and detections
labels = [item["class"] for item in result["predictions"]]
detections = sv.Detections.from_roboflow(result)

# Load the image
image = cv2.imread("D:/Proj/Conjuctiva/istockphoto-1388917616-612x612.jpg")

# Annotate bounding boxes
bounding_box_annotator = sv.BoundingBoxAnnotator()
annotated_frame = bounding_box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)

# Annotate labels
label_annotator = sv.LabelAnnotator()
annotated_image = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

# Extract and save only the region inside the bounding box
i = 0
for detection in detections:
    coordinates = detection[0]
    
    x, y, w, h = map(int, coordinates)
    
    # Create a mask with the same dimensions as the image
    mask = np.zeros_like(image)
    
    # Fill the bounding box area on the mask with white
    mask[y:h, x:w] = image[y:h, x:w]
    
    # Extract the region of interest (ROI) from the masked image
    roi = mask[y:h, x:w]
    
    # Display the ROI
    sv.plot_image(image=roi, size=(16, 16))
    
    # Save the ROI as needed
    cv2.imwrite(f'D:/Proj/Conjuctiva/conjuctiva_region_{i}.png', roi)
    i += 1

from tensorflow.keras.models import load_model

def predict_anemia(model, img_path, image_size=(128, 128)):
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, image_size)
        img = img / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make a prediction
        prediction = model.predict(img_array)

        # Convert the prediction to a binary class label
        threshold = 0.5
        class_label = (prediction > threshold).astype(int)

        # Output the result
        if class_label == 0:
            return "Not Anemic"
        else:
            return "Anemic"
    else:
        return "Image could not be loaded"

# Example usage:

img_path = r'D:/Proj/Conjuctiva/conjuctiva_region_0.png';  # Update this path
model_path=r'conjunctiva_100.h5';

fingerNail__model = load_model(model_path)

# Predict using the model
result = predict_anemia(fingerNail__model, img_path)
print(result)
