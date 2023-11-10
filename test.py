from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import numpy as np
import cv2
import os

model = fasterrcnn_resnet50_fpn(pretrained=True)
# model = ssdlite320_mobilenet_v2(pretrained=True) 
model.eval()

transform = T.Compose([T.ToTensor()])

image_file = 'test/image8.jpg'
image = Image.open(image_file)
image.show()

image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)  

with torch.no_grad():
    prediction = model(image_tensor)

boxes = prediction[0]['boxes'].cpu().numpy()
labels = prediction[0]['labels'].cpu().numpy()
car_coordinates = []

# IMG_1992.JPG.4emlqlh4.ingestion-6db97b7d4c-ncthg
# IMG_1992.JPG.4emlqlh4.ingestion-6db97b7d4c-ncthg

for i in range(len(boxes)):
    print('labels', labels[i])
    if labels[i] == 3 or labels[i] == 6: 
    # or labels[i] == 77 or labels[i] == 36 or labels[i] == 41 or labels[i] == 75 or labels[i] == 8: 
        print('boxes', boxes[i])
    # if labels[i] == 3: 
        # print('boxes', boxes[i])
        x, y, w, h = boxes[i]
        x_center = x + w/2
        y_center = y + h/2
        car_coordinates.append((x, y, w, h))

print(f"Car coordinates in img{1}.jpg: {car_coordinates}")

# image_array = cv2.imread("output2.jpg")

# # Draw non-parking coordinates as red rectangles
# for coord in car_coordinates:
#     x, y, w, h = coord
#     x, y, w, h = int(x), int(y), int(w), int(h)
#     width, height = 500, 500  
#     radius = 50
#     cv2.rectangle(image_array, (x, y), (x + width, y + height), (0, 0, 255), 2)  # Red rectangle
#     # cv2.circle(image_array, (x, y), radius, (0, 0, 255), 2)

# cv2.imshow("Parking Coordinates", image_array)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# image.show()