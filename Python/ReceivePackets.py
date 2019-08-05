from socket import socket, gethostbyname, AF_INET, SOCK_DGRAM
import sys
import struct
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import pyRigolDP832 as dp


PORT_NUMBER = 7000
SIZE = 1024

hostName = gethostbyname('0.0.0.0')

mySocket = socket(AF_INET, SOCK_DGRAM)
mySocket.bind((hostName, PORT_NUMBER))

print("Test server listening on port {0}\n".format(PORT_NUMBER))
pixel_list = []
try:
    for i in range(0, 20):
        pixel_list.append(i)
    while True:
        (data, addr) = mySocket.recvfrom(SIZE)
        data_arr = struct.unpack("493h", data)
        pixel_list.append(data_arr)
        if data_arr[0] == 1:
            print("Received " + str(len(pixel_list)) + " packets.")
            break
except KeyboardInterrupt:
    print("Process interrupted.")

img = np.zeros((3648, 5472))
pixel_list = pixel_list[20:]
for i, image_segment in enumerate(pixel_list):
    image_segment = image_segment[1:len(image_segment)]
    for j, val in enumerate(image_segment):
        if j > len(image_segment) - 3:
            break
        if j % 3 == 0:
            pixel_val = image_segment[j]
            pixel_x = image_segment[j + 1]
            pixel_y = image_segment[j + 2]
            img[pixel_x][pixel_y] = pixel_val

plt.imshow(img)
plt.show()

sys.exit()
