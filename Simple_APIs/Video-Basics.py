import cv2
import numpy as np

# Create a function based on a CV2 Event (Left button click)
def draw_circle(event, x, y, flags, param):
    global center, clicked

    # get mouse click on down and track center
    if event == cv2.EVENT_LBUTTONDOWN:
        center = (x, y)
        clicked = True

# Haven't drawn anything yet!
center = (0, 0)
clicked = False

# Create a named window for connections
cv2.namedWindow('Video')

# Capture Video
cap = cv2.VideoCapture(0)  # Change this to your video source, e.g., 'video.mp4'

# Bind draw_circle function to mouse clicks
cv2.setMouseCallback('Video', draw_circle)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Use if statement to see if clicked is true
    if clicked:
        # Draw a blue circle on the frame (unfilled)
        cv2.circle(frame, center, 20, (255, 0, 0), 2)
        
    # Display the resulting frame
    cv2.imshow('Video', frame)

    # This command lets us quit with the "q" button on a keyboard.
    # Simply pressing X on the window won't work!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close the window
cap.release()
cv2.destroyAllWindows()

