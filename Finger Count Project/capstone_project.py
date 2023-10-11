import cv2
import numpy as np
# Used for distance calculation
from sklearn.metrics import pairwise

### GLOBAL VARIABLES ###
background = None
accumulated_weight = 0.5
# Region of Intrest, rectangle, depends on camera window
roi_top = 80
roi_bottom = 330
roi_right = 20
roi_left = 270

### FUNCTIONS ###
def calculate_accumulated_average(frame,accumulated_weight):
    
    global background
    
    # First time make a copy of the frame
    if background is None:
        background = frame.copy().astype('float')
        return None
    
    # Update background with accumulated weight
    cv2.accumulateWeighted(frame,background,accumulated_weight) 
    
# Adjust threshold accordingly
def segmentation(frame,threshold=25):
    
    diff = cv2.absdiff(background.astype('uint8'),frame)
    
    # return, thresh_image, to get contrasts
    ret, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Retrieve only the external contours (contours on the outer edges of objects) 
    # cv2.CHAIN_APPROX_SIMPLE contour approximation method that compresses horizontal, vertical, and diagonal segments and leaves only their end points. 
    # It reduces the number of points and simplifies the representation of the contour. 
    image, contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours could be retrieved
    if len(contours) == 0:
        return None
    else: 
        # Assuming the largest external contour in ROI is the hand
        hand_segment = max(contours,key=cv2.contourArea)
        
        return (thresh,hand_segment)

def count_fingers(thresh,hand_segment):
    
    # Convex polygon
    conv_hull = cv2.convexHull(hand_segment)
    
    # Grab extreme points, top, bottom, left right
    top    = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left   = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right  = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])
    
    # Center of hand
    center_X = int((left[0] + right[0]) / 2)
    center_Y = int((top[1] + bottom[1]) / 2)
    
    # Euclidean distance
    distance = pairwise.euclidean_distances([(center_X, center_Y)], Y=[left, right, top, bottom])[0]
    
    # Point furthest away
    max_distance = distance.max()
    
    # Create circle, percent depends on persons hand
    # If short fingers, take less percent
    percent = 0.8
    radius = int(percent*max_distance)
    circumference = (2*np.pi*radius)
    
    # Empty black image with the same dimensions as the thresholded image. All pixel values set to 0 (black)
    circular_roi = np.zeros(thresh.shape[:2],dtype='uint8')
    
    # Draw white circle with thickness 10 on the black circular_roi image
    cv2.circle(circular_roi,(center_X,center_Y),radius,255,10)
    
    # Masks out the circular region from the original thresholded image, keeping only the portion that falls within the circular ROI. Pixels outside the circle become black (0)
    circular_roi = cv2.bitwise_and(thresh, thresh, mask=circular_roi)

    # Grab contours in the cirular_roi
    image, contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Finger count starts at 0, no fingers extended
    finger_count = 0

    # loop through the contours to see if we count any more fingers.
    for count in contours:
        
        # Bounding box of countour
        (x, y, w, h) = cv2.boundingRect(count)

        # Increment count of fingers based on two conditions:
        percent = 0.25
        # 1. Contour region is not the very bottom of hand area (the wrist)
        out_of_wrist = ((center_Y + (center_Y * 0.25)) > (y + h))
        
        # 2. Number of points along the contour does not exceed certain % of the circumference of the circular ROI (otherwise we're counting points off the hand)
        limit_points = ((circumference * percent) > count.shape[0])
        
        
        if  out_of_wrist and limit_points:
            finger_count += 1

    return finger_count

### PROGRAM ###

# Start camera
cam = cv2.VideoCapture(0)

# Intialize a frame count
num_frames = 0

# keep looping, until interrupted
while True:
    # Get the current frame
    ret, frame = cam.read()

    # Flip the frame so that it is not the mirror view
    frame = cv2.flip(frame, 1)

    # Copy the frame
    frame_copy = frame.copy()

    # Grab the Region of Intrest from the frame
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]
    
    # Image is in BGR
    # Apply grayscale and blur to roi image
    gray_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 7x7 Kernel
    gray_img = cv2.GaussianBlur(gray_img, (7, 7), 0)

    # For the first 60 frames we will calculate the average of the background.
    # Give message to the user
    if num_frames < 60:
        calculate_accumulated_average(gray_img, accumulated_weight)
        if num_frames <= 59:
            cv2.putText(frame_copy, "Getting Background now, please wait.", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Finger Count",frame_copy)
            
    else:
        # Now that we have the background, we can segment the hand.        
        # Segment the hand region
        hand = segmentation(gray_img)

        # Check if we could detect a hand
        if hand is not None:
            
            # Unpack hand
            thresh, hand_segment = hand

            # Draw contours around the hand
            cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0),1)

            # Count how many fingers are shown
            num_fingers = count_fingers(thresh, hand_segment)

            # Display finger counts on the screen
            cv2.putText(frame_copy, str(num_fingers), (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # Display the thresholded image
            cv2.imshow("Thesholded", thresh)

    # Draw ROI Rectangle on frame copy
    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 5)

    # Increment the number of frames for tracking
    num_frames += 1

    # Display the frame with segmented hand
    cv2.imshow("Finger Count", frame_copy)


    # Close windows with Esc
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

# Release the camera and destroy all the windows
cam.release()
cv2.destroyAllWindows()

