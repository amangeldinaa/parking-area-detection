import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import numpy as np
import cv2
import os

error_range = 40

def extract_coordinates():
    model = fasterrcnn_resnet50_fpn(pretrained=True) 
    model.eval()

    # folder_path = "test_images2"
    # image_files = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg", "img6.jpg"]
    # image_files = ["img1-2.jpg", "img2-2.jpg", "img3-2.jpg", "img4-2.jpg", "img5-2.jpg", "img6-2.jpg"]
    # image_files = ["img1-3.jpg", "img2-3.jpg", "img3-3.jpg", "img4-3.jpg", "img5-3.jpg", "img6-3.jpg"]
    # image_files = [filename for filename in os.listdir(folder_path) if filename.endswith('.jpg')]

    folder_path = "processed-data"
    # folder_path = "test"
    image_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.jpg')]
    print('image_files',image_files)

    transform = T.Compose([T.ToTensor()])
    image_inputs = []

    for image_file in image_files:
        image = Image.open(image_file)
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  

        with torch.no_grad():
            prediction = model(image_tensor)

        boxes = prediction[0]['boxes'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        car_coordinates = []

        for i in range(len(boxes)):
            if labels[i] == 3 or labels[i] == 6:
            # 77 or labels[i] == 36 or labels[i] == 41 or labels[i] == 75: 
                print('boxes', boxes[i])
            # if labels[i] == 3: 
                # print('boxes', boxes[i])
                x, y, w, h = boxes[i]
                x_center = x + w/2
                y_center = y + h/2
                car_coordinates.append((x, y, w, h))

        image_inputs.append([tuple(coord[:2]) for coord in car_coordinates])

    for i, car_coordinates in enumerate(image_inputs):
        print(f"Car coordinates in img{i + 1}.jpg: {car_coordinates}")
    return image_inputs


def isSameArea(coord1, coord2):
    x_diff = abs(coord1[0] - coord2[0])
    y_diff = abs(coord1[1] - coord2[1])
    if x_diff <= error_range and y_diff <= error_range:
        return True
    else:
        return False


def get_bounding_box(coordinates):
    if not coordinates:
        return None

    min_x = min(coord[0] for coord in coordinates)
    min_y = min(coord[1] for coord in coordinates)
    max_x = max(coord[0] for coord in coordinates)
    max_y = max(coord[1] for coord in coordinates)

    return [(min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)]


def main():
    parking_coordinates = []
    nonparking_coordinates = []
    image_inputs = extract_coordinates()
    print('image_inputs', image_inputs)
    sorted_inputs = [sorted(group, key=lambda coord: coord[0]) for group in image_inputs]
    print('sorted_inputs', sorted_inputs)

    for i in range(0, len(sorted_inputs), 3):
        group = sorted_inputs[i:i+3]
        for coord in group[0]:
            added = False
            for coord2 in group[1]:
                if(isSameArea(coord, coord2)):
                    parking_coordinates.extend([coord, coord2])
                    for coord3 in group[2]:
                        if(isSameArea(coord, coord3) and isSameArea(coord2, coord3)): 
                            # x_average = (coord[0] + coord2[0] + coord3[0]) / 3
                            # y_average = (coord[1] + coord2[1] + coord3[1]) / 3
                            # parking_coordinates.extend([(x_average, y_average)])
                            parking_coordinates.extend([coord, coord2, coord3])
                            added = True
                        else:  
                            if(coord3 not in parking_coordinates):
                                nonparking_coordinates.extend([coord3])
                else:
                    if(coord2 not in parking_coordinates):
                        nonparking_coordinates.extend([coord2])
                    for coord3 in group[2]:
                        if(isSameArea(coord2, coord3)):
                            # x_average = (coord2[0] + coord3[0]) / 2
                            # y_average = (coord2[1] + coord3[1]) / 2
                            # parking_coordinates.extend([(x_average, y_average)])
                            parking_coordinates.extend([coord2, coord3])
            if(added == False and coord not in parking_coordinates):
                nonparking_coordinates.extend([coord])

    parking_set = set(parking_coordinates)
    # Remove tuples from nonparking_coordinates that are present in parking_coordinates
    nonparking_coordinates = [coord for coord in nonparking_coordinates if coord not in parking_set]
    parking_coordinates = list(parking_set)
    nonparking_coordinates = list(set(nonparking_coordinates))

    print("\nParking Coordinates:", parking_coordinates)
    print("Non-Parking Coordinates:", nonparking_coordinates)
    print('---------------------\n')

    # change to "img1-2.jpg" or "img1-3.jpg" when testing on other images
    image_array = cv2.imread("test.jpg")

    # Draw parking coordinates as green circles
    for coord in parking_coordinates:
        x, y = coord
        x, y = int(x), int(y)
        width, height = 500, 500  
        cv2.rectangle(image_array, (x, y), (x + width, y + height), (0, 255, 0), 2)  # Green rectangle

    # Draw non-parking coordinates as red rectangles
    for coord in nonparking_coordinates:
        x, y = coord
        x, y = int(x), int(y)
        width, height = 500, 500  
        # radius = 50
        cv2.rectangle(image_array, (x, y), (x + width, y + height), (0, 0, 255), 2)  # Red rectangle
        # cv2.circle(image_array, (x, y), radius, (0, 0, 255), 2)
    
    non_parking_bbox = get_bounding_box(nonparking_coordinates)
    print('non_parking_bbox', non_parking_bbox)
    # for coord in non_parking_bbox:
    #     x, y = coord
    #     x, y = int(x), int(y)
    #     width, height = 10,10  
    #     cv2.rectangle(image_array, (x, y), (x + width, y + height), (255, 255, 0), 2)  # blue rectangle

    # if len(non_parking_bbox) == 4:
    #     rect_color = (255, 255, 0)  # Blue color
    #     rect_thickness = 2

    #     # Convert the coordinates to integers
    #     pt1 = (int(non_parking_bbox[0][0]), int(non_parking_bbox[0][1]))
    #     pt2 = (int(non_parking_bbox[2][0]), int(non_parking_bbox[2][1]))

    # # Draw the rectangle
    # cv2.rectangle(image_array, pt1, pt2, rect_color, rect_thickness)

    cv2.imshow("Parking Coordinates", image_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()