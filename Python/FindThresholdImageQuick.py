import numpy as np
import cv2
import sys
import math
import time

save_folder = "/home/schoen/Documents/ImageProcessPython/Images/" # Initialize save path

image = cv2.imread(save_folder + "Picture 1.jpg") # Read in sample image
print(image.shape) #Print it's shape

width = image.shape[1] #Find width, height of image
height = image.shape[0]
sigma_multiple = 3 # number of standard deviations for thresholding

img_dev = np.empty((height, width)) # Create empty array to hold threshold values
row = 0
pixel = 0
images = [] # Create array to hold images\
num_images = 10 #Find number of images you want to threshold with
for i in range(0, num_images):
	temp_img = cv2.imread(save_folder + "Picture " + str(i + 1) + ".jpg") # Read in each image
	images.append(temp_img) # Add each image to images array
	print("Adding image " + str(i))
times = []

try:
	for i in range(0, height - 1): #For each row
		print("Col: " + str(i))
		t0 = time.time()
		for j in range(0, width - 1): # For each pixel in that row
			img_data = np.empty(num_images) # Create empty array to hold pixel values for each pixel
			val = 0
			for pic in images: #For each picture in images, 
				img_data[val] = pic[i, j, 0] #Add value to image data array
				val += 1 # increment index
			# Set threshold to the mean value plus the deviation
			standard_dev = img_data.std() * sigma_multiple
			mean = img_data.mean()
			img_dev[i,j] = mean + standard_dev # Set threshold of each pixel to mean pixel value plus 3 std. deviations
		t1 = time.time()
		times.append(t1 - t0)

except KeyboardInterrupt:
	print("Program terminated.")

total_time = 0
for time in times:
	total_time += time


average_time = total_time / len(times)
print("Average time per row: " + str(average_time) + "s")
print("Total time to run: " + str((average_time * width) / 60) + "min")
print(img_dev)
cv2.imwrite(save_folder + "Threshold.jpg", img_dev)
