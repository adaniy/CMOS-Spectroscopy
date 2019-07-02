import numpy as np
import cv2
import time


t0 = time.time()
save_folder = "C:\\Users\\habraha1\\Documents\\Images\\"
print(cv2.__version__)
image = cv2.imread(save_folder + "Picture 1.jpg")
print(image.shape)
width = image.shape[1]
height = image.shape[0]
sigma_multiple = 3
img_dev = np.empty((height, width))
NUM_IMAGES = 200
images = np.empty((NUM_IMAGES, height, width), dtype="uint8")
for i in range(0, NUM_IMAGES):
    temp_img = cv2.imread(save_folder + "Picture " + str(i) + ".jpg")
    temp_img = temp_img[:, :, 0]
    images[i] = temp_img
    print("Adding image " + str(i) + "...")
print("Images shape: " + str(images.shape))
try:
    print("Finding variance...")
    y = np.var(images, axis=0, dtype=np.dtype("uint16"))
    print("Finding square roots...")
    y = np.sqrt(y, dtype="float32")
    print("Calculating standard deviations...")
    img_dev = np.mean(images, axis=0) + (y * 3)

except KeyboardInterrupt:
    print("Program terminated.")

t1 = time.time()

print("Total time to run: " + str(t1 - t0))
print(img_dev)
print(img_dev.shape)
cv2.imwrite(save_folder + "Threshold.jpg", img_dev)
