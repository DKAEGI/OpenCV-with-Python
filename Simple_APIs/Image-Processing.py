import cv2
import matplotlib.pyplot as plt
import numpy as np

def display_img(img,cmap=None):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)

giraffes = cv2.imread('../DATA/giraffes.jpg') # Original BGR for OpenCV --> Here Image is loaded as a non gray scale image
show_giraffes = cv2.cvtColor(giraffes, cv2.COLOR_BGR2RGB) # Converted to RGB for Matplotlib
display_img(show_giraffes)

###Apply a binary threshold onto the image.

# Non gray scale image
ret,thresh1 = cv2.threshold(giraffes,127,255,cv2.THRESH_BINARY)
# RESULT IS NOT GOOD WITH A NON GRAY SCALE IMAGE
display_img(thresh1,cmap='gray')
# LOAD IMAGE AS A GRAYSCALE IMAGE, ADD 0 TO IT!!!
giraffes = cv2.imread('../DATA/giraffes.jpg',0) # Original BGR for OpenCV
ret,thresh1 = cv2.threshold(giraffes,127,255,cv2.THRESH_BINARY)
display_img(thresh1,cmap='gray')

### Open the giaraffes.jpg file from the DATA folder and convert its colorspace to  HSV and display the image
giraffes = cv2.imread('../DATA/giraffes.jpg') # Original BGR for OpenCV --> Here Image is loaded as a non gray scale image --> But is needed to be able to convert the color space
show_giraffes_HSV = cv2.cvtColor(giraffes, cv2.COLOR_BGR2HSV) # Converted to HSV
display_img(show_giraffes_HSV)

### Create a low pass filter with a 4 by 4 Kernel filled with values of 1/10 (0.01) and then use 2-D Convolution to blur the giraffer image (displayed in normal RGB)
kernel = np.ones((4,4),np.float32)/10
kernel
giraffes = cv2.imread('../DATA/giraffes.jpg') # Original BGR for OpenCV --> Here Image is loaded as a non gray scale image
show_giraffes = cv2.cvtColor(giraffes, cv2.COLOR_BGR2RGB) # Converted to RGB for Matplotlib
# Apply 2D convolution to blur the image
blurred_image = cv2.filter2D(show_giraffes, -1, kernel)
display_img(blurred_image)

### Create a Horizontal Sobel Filter (sobelx from our lecture) with a kernel size of 5 to the grayscale version of the giaraffes image and then display the resulting gradient filtered version of the image.
# LOAD IMAGE AS A GRAYSCALE IMAGE, ADD 0 TO IT!!!
giraffes = cv2.imread('../DATA/giraffes.jpg',0) # Original BGR for OpenCV

sobelx = cv2.Sobel(giraffes,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(giraffes,cv2.CV_64F,0,1,ksize=5)
laplacian = cv2.Laplacian(giraffes,cv2.CV_64F)

display_img(sobelx,cmap='gray')
# Plot the color histograms for the RED, BLUE, and GREEN channel of the giaraffe image. Pay careful attention to the ordering of the channels.**
giraffes = cv2.imread('../DATA/giraffes.jpg') # Original BGR for OpenCV
img = giraffes
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.title('Giraffes Image')
plt.show()

