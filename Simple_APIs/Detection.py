import numpy as np
import cv2 
import matplotlib.pyplot as plt

### Read in the car_plate.jpg file from the DATA folder.
# Open image in grayscale
car_plate = cv2.imread('../DATA/car_plate.jpg')


### Create a function that displays the image in a larger scale and correct coloring for matplotlib.
def display(img):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    # Convert to RGB
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(new_img)

display(car_plate)


###Load the haarcascade_russian_plate_number.xml file.
plate_cascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_russian_plate_number.xml')


###Create a function that takes in an image and draws a rectangle around what it detects to be a license plate. Keep in mind we're just drawing a rectangle around it for now, later on we'll adjust this function to blur. You may want to play with the scaleFactor and minNeighbor numbers to get good results.
def detect_plate(img):
    
  
    plate_img = img.copy()
  
    plate_rects = plate_cascade.detectMultiScale(plate_img,scaleFactor=1.2, minNeighbors=5)
    
    for (x,y,w,h) in plate_rects: 
        cv2.rectangle(plate_img, (x,y), (x+w,y+h), (0,0,255), 3) 
        
    return plate_img

result = detect_plate(car_plate)

display(result)

### Edit the function so that is effectively blurs the detected plate, instead of just drawing a rectangle around it. Here are the steps you might want to take:
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
