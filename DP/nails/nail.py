from roboflow import Roboflow
import supervision as sv
import cv2
from PIL import Image

rf = Roboflow(api_key="tnmXe3ZHhmDuk99BH57W")
project = rf.workspace().project("nakhoon")
model = project.version(1).model
im = Image.open("./pexels-kristina-paukshtite-704815-removebg-preview (1).png")
im = im.convert("RGB")
print(im.mode)
im.save("D:/Proj/nails/nail.png", "PNG", quality=95)
result = model.predict("D:/Proj/nails/nail.png", confidence=40).json()

# labels = [item["class"] for item in result["predictions"]]

# print(result)

detections = sv.Detections.from_roboflow(result)

image = cv2.imread("D:/Proj/nails/pexels-kristina-paukshtite-704815-removebg-preview (1).png")
bounding_box_annotator = sv.BoundingBoxAnnotator()
annotated_frame = bounding_box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
# for detection in detections:
#     # print("Detection:", detection)
#     coordinates = detection[0]
#     print("x=",detection[0][0])
#     print("y=",detection[0][1])
#     print("width=",detection[0][2])
#     print("height=",detection[0][3])
#     # Extract coordinates
#     confidence_score = detection[2]
#     print(coordinates)
#     print(confidence_score)


label_annotator = sv.LabelAnnotator()
# mask_annotator = sv.MaskAnnotator()



# annotated_image = mask_annotator.annotate(
#     scene=image, detections=detections)
annotated_image = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)
i=0
sv.plot_image(image=annotated_image, size=(16, 16))
for detection in detections:
    coordinates = detection[0]
    
    x, y, w, h = map(int, coordinates)
    roi = annotated_frame[y:h, x:w]
    sv.plot_image(image=roi, size=(16, 16))
    
    
    # Save or process the ROI as needed
    cv2.imwrite(f'D:/Proj/nails/palm_region_{i}.png', roi)
    i=i+1
   
