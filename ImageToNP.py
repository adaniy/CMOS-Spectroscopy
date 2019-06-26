import PySpin, time, datetime
import numpy as np
from numba import *
import os

save_folder = "/home/schoen/Documents/Blackfly_Images" # Change later
exposure_time = 200 # in milliseconds
capture_fps = 4.9
gain = 0 # ISO for digital cameras
reverse_x = False
reverse_y = False
bit = 8

if not os.path.exists(save_folder):
	os.mkdir(save_folder)

# Get PySpin system
system = PySpin.System.GetInstance()

# Get camera list
cam_list = system.GetCameras()
cam = cam_list.GetByIndex(0)
cam.Init()

print("Camera firmware version: " + cam.DeviceFirmwareVersion.ToString())

#load default config
cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
cam.UserSetLoad()

# set acquisition to continuous, turn off auto exposure, set the frame rate
#Camera Settings
#Set Packet Size
cam.GevSCPSPacketSize.SetValue(9000)
cam.DeviceLinkThroughputLimit.SetValue(100000000)
#Set acquistion mode
cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
#Set exposure time
cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
cam.ExposureMode.SetValue(PySpin.ExposureMode_Timed)
cam.ExposureTime.SetValue(exposure_time * 1e3)
#Set FPS
cam.AcquisitionFrameRateEnable.SetValue(True)
cam.AcquisitionFrameRate.SetValue(capture_fps)
#set analog, gain, turn off gamma
cam.GainAuto.SetValue(PySpin.GainAuto_Off)
cam.Gain.SetValue(gain)
cam.GammaEnable.SetValue(False)

cam.ReverseX.SetValue(reverse_x)
cam.ReverseY.SetValue(reverse_y)
if bit > 8:
	image_bit = 16
	cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit12)
	cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono12p)
else:
	image_bit = 8
	cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit10)
	cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)


def process_np(array):
	if array > 127:
		array = 255
	else:
		array = 0
	return array

def save_image(image, val):
	#convert PySpin image object into an ND array, rescale, save jpg image and nd array\
	time_str = val
	if image_bit == 16:
		img_nd = np.copy(image.Convert(PySpin.PixelFormat_Mono16).GetNDArray())
	else:
		img_nd = np.copy(image.GetNDArray())
	if image.IsIncomplete():
		print('Image incomplete with image status %s...' % image.GetImageStatus())
		print(PySpin.Image_GetImageStatusDescription(image.GetImageStatus()))
	else:
		# Print image info
		print('Grabbed image %i, width = %i, height = %i' % (val,image.GetWidth(),image.GetHeight()))
	print(img_nd)
	print("Processing...")
	t1 = time.time()
	vectorized_process = vectorize(["uint8(uint8)"], target = "cuda")(process_np)
	vectorized_process(img_nd) #Recheck rules for @vectorize
	t2 = time.time()
	times.append(t2-t1)
	print("Time to process: " + str(t2 -t1))
	print(img_nd)
	#image.Save("{}/{}.jpg".format(save_folder,val))
	#np.save("{}/{}".format(save_folder, time_str), img_nd)
	return img_nd
i=0
cam.BeginAcquisition()
save_threads = []
times = []
input("Press enter to start. Press ctrl+c to stop.")
t3 = time.time()
try:
	while True:
		image = cam.GetNextImage()
		save_image(image, i)
		#save_threads.append(threading.Thread(target=save_image, args=(image,i,)))
		#save_threads[-1].start()
		print(str(i))
		i += 1
except KeyboardInterrupt:
	t0 = time.time()
			

total_time = 0
for t in times:
	total_time += t

average_time = total_time / len(times)

print("Average processing time: " + str(average_time))
fps = i / (t0 -t3)
print("Frames per second (actual): " + str(fps))

#for thread in save_threads:
	#thread.join()

print("Capturing time: {:.3f}s".format(t0-t3))
print("Acquisition resulting frame rate: {:.2f}FPS".format(cam.AcquisitionResultingFrameRate()))

cam.EndAcquisition()
cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
cam.UserSetLoad()
cam.DeInit()
cam_list.Clear()
del image
del cam
del cam_list
system.ReleaseInstance()


