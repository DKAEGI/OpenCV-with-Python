import numpy as np
import cv2 
import matplotlib.pyplot as plt

### Read in the car_plate.jpg file from the DATA folder.
# Open image in grayscale
car_plate = cv2.imread('../DATA/car_plate.jpg')


### Function that displays the image in a larger scale and correct coloring for matplotlib.
def display(img):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    # Convert to RGB
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(new_img)

display(car_plate)


### Load the haarcascade_russian_plate_number.xml file.
plate_cascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_russian_plate_number.xml')


### Function that takes in an image and draws a rectangle around what it detects to be a license plate. 
def detect_plate(img):
    
  
    plate_img = img.copy()
  
    plate_rects = plate_cascade.detectMultiScale(plate_img,scaleFactor=1.2, minNeighbors=5)
    
    for (x,y,w,h) in plate_rects: 
        cv2.rectangle(plate_img, (x,y), (x+w,y+h), (0,0,255), 3) 
        
    return plate_img

result = detect_plate(car_plate)

display(result)

### Function which blurs the detected plate.
def detect_and_blur_plate(img):
    
    plate_img = img.copy()
    
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.2, minNeighbors=5)
    
    for (x, y, w, h) in plate_rects:
        # Extract the license plate region
        plate_roi = plate_img[y:y+h, x:x+w]
        
        # Apply median blur to the license plate region
        blurred_plate = cv2.medianBlur(plate_roi, 7)  # You can adjust the kernel size (e.g., 7) for stronger/weaker blur
        
        # Replace the original license plate region with the blurred version
        plate_img[y:y+h, x:x+w] = blurred_plate
        
    return plate_img

result = detect_and_blur_plate(car_plate)

display(result)
