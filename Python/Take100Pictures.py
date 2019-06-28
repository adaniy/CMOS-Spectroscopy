import PySpin, time, datetime
import numpy as np
from numba import *
import os
import multiprocessing

save_folder = "/home/schoen/Documents/ImageProcessPython/Images/" # Change later
exposure_time = 200 # in milliseconds
capture_fps = 4.9
gain = 0 # ISO for digital cameras
reverse_x = False
reverse_y = False
bit = 8
numImages = 100
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
if bit > 8: #If requested bit level is above 8, make it 16
	image_bit = 16 
	cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit12)
	cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono12p)
else: #Otherwide, make it 8
	image_bit = 8
	cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit10)
	cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
	
	
	
cam.BeginAcquisition()
print("Beginning acquisition...")
try:
	for i in range(0, numImages):
		image = cam.GetNextImage()
		if (image.IsIncomplete()):
			print("Image " + str(i + 1) + " is incomplete.")
		image.Save(save_folder + "Picture " + str(i) + ".jpg")
		print("Captured image: " + str(i + 1))
except KeyboardInterrupt:
	print("Process Interrupted.")
#Clean up 
cam.EndAcquisition()
cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
cam.UserSetLoad()
cam.DeInit()
cam_list.Clear()
del image
del cam
del cam_list
system.ReleaseInstance()









