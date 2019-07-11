"""CMOSProcess.py: Pulls images from FLIR CMOS Cameras and processes them for Plasma Spectroscopy."""

__author__ = "Hunter Abraham"
__credits__ = ["Adam Schoenwald"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Hunter Abraham"
__email__ = "hjabraham@wisc.edu"
__status__ = "Production"

import numpy as np
import argparse
import datetime
import PySpin
import cv2
import time
import multiprocessing
import os
import bokeh


class Process():
    save_jpg = False  # Instance fields to determine what "path" code will take
    save_np = False
    threshold = False
    find_threshold_bool = -1
    multi = False
    num_images = -1
    save_folder = "Images/"
    system = None
    std = -1

    def __init__(self, save_jpg, save_np, threshold, num_images, find_threshold_bool, multi, std):
        
        self.save_jpg = save_jpg
        self.save_np = save_np
        self.threshold = threshold
        self.num_images = num_images
        self.find_threshold_bool = find_threshold_bool
        self.multi = multi
        self.std = std
        
    def find_threshold(self, image_threshold, num_images):
        t0 = time.time()
        width = image_threshold.shape[1]
        height = image_threshold.shape[0]
        img_dev = np.empty((height, width))
        NUM_IMAGES = num_images
        images_threshold = np.empty((NUM_IMAGES, height, width), dtype="float64")
        for i in range(0, NUM_IMAGES):
            temp_img = cam.GetNextImage()
            temp_img = temp_img.GetNDArray()
            images_threshold[i] = temp_img
        try:
            print("Thresholding...")
            y = np.var(images_threshold, axis=0, dtype=np.dtype("float64"))
            y = np.sqrt(y, dtype="float64")
            img_dev = np.mean(images_threshold, axis=0) + (y * self.std)
        except KeyboardInterrupt:
            print("Program terminated.")
        t1 = time.time()
        np.clip(img_dev, 0, 255)
        cv2.imwrite("Images/Threshold.tiff", img_dev)
        print("Done thresholding")
        return img_dev

    def convert_images(self, temp_image_to_be_thresholded, temp_threshold, pic_num):
        temp_image_to_be_thresholded[temp_image_to_be_thresholded > temp_threshold] = 255
        temp_image_to_be_thresholded[temp_image_to_be_thresholded <= temp_threshold] = 0
        if self.save_jpg:  # If user wants to save image as .jpg, save as .jpg
            cv2.imwrite(self.save_folder + 'Processed Picture' + str(pic_num) + '.tiff', temp_image_to_be_thresholded)
        if self.save_np:  # If user wants to save image as .npp, save as .npp
            np.save(self.save_folder + "Processed Array " + str(pic_num), temp_image_to_be_thresholded)

    def setup(self):
        exposure_time = 200  # in milliseconds
        capture_fps = 4.9
        gain = 0  # ISO for digital cameras
        reverse_x = False
        reverse_y = False
        bit = 8
        # Get PySpin system
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)
        # load default config
        cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
        cam.UserSetLoad()
        # set acquisition to continuous, turn off auto exposure, set the frame rate
        # Camera Settings
        # Set Packet Size
        cam.GevSCPSPacketSize.SetValue(9000)
        cam.DeviceLinkThroughputLimit.SetValue(125000000)
        # Set acquistion mode
        cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        # Set exposure time
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        cam.ExposureMode.SetValue(PySpin.ExposureMode_Timed)
        cam.ExposureTime.SetValue(exposure_time * 1e3)
        # Set FPS
        cam.AcquisitionFrameRateEnable.SetValue(True)
        cam.AcquisitionFrameRate.SetValue(capture_fps)
        # set analog, gain, turn off gamma
        cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        cam.Gain.SetValue(gain)
        cam.GammaEnable.SetValue(False)
        cam.ReverseX.SetValue(reverse_x)
        cam.ReverseY.SetValue(reverse_y)
        if bit > 8:  # If requested bit level is above 8, make it 16
            image_bit = 16
            cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit12)
            cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono12p)
        else:  # Otherwise, make it 8
            image_bit = 8
            cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit10)
            cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
        cam.BeginAcquisition()

    def whole_capture(self, cam):
        self.setup()
        processes = []
        threshold_img = None
        i = 0
        try:

            while True:  # Take n images
                if i == self.num_images:
                    break
                if i == 0:
                    image_first = cam.GetNextImage()
                    image_np_first = image_first.GetNDArray().astype("uint32")
                    print(image_np_first.dtype)
                    print(image_np_first.shape)
                if self.find_threshold_bool != -1:  # If the user needs to find a threshold image,
                    threshold_img = self.find_threshold(image_np_first,
                                                        self.find_threshold_bool)  # Find a threshold image
                    self.find_threshold_bool = -1  # Make sure it only goes through this process once as it is expensive
                else:
                    threshold_img = cv2.imread(
                        "Images/Threshold.tiff")  # If they don't want to find a new threshold, pull the old one
                    threshold_img = threshold_img[:, :, 0]
                print("Capturing image " + str(i))
                image = cam.GetNextImage()
                if image.IsIncomplete():
                    print("Error: image is incomplete.")
                    continue
                image_np = image.GetNDArray()  # Convert image to nd array
                if self.save_jpg:  # If user wants to save image as .tiff, save as .tiff
                    cv2.imwrite(self.save_folder + 'Unprocessed Picture ' + str(i) + '.tiff',
                                image_np)
                if self.save_np:  # If user wants to save image as .npy, save as .npy
                    np.save(self.save_folder + "Unprocessed Array " + str(i), image_np)
                if self.threshold:  # If they want thresholding,
                    if self.multi:  # If they want multiprocessing,
                        multiprocessing_threshold_image = multiprocessing.Process(target=self.convert_images, args=(
                            image_np, threshold_img, i,))  # Create a call to multiprocessing
                        processes.append(multiprocessing_threshold_image)  # Append it to list of processes
                        multiprocessing_threshold_image.start()  # Start process
                    else:
                        self.convert_images(image_np, threshold_img, i)  # Otherwise, process in standard format
               
                del image
                image_np = None
                i += 1
        except KeyboardInterrupt:  # If keyboard interrupt (ctrl+c) is found, kill loop and print message
            print('Process interrupted.')

        if self.multi:  # If multiprocessing was used, clean up
            for process_temp in processes:
                process_temp.join()

       

        print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parses arguments")  # Create argument parser
    parser.add_argument('--num', type=int,
                        help='number of images to collect')  # Positional argument to specify number of images
    parser.add_argument('--jpg', default=False, type=lambda x: (str(x).lower() == 'true'))  # FIXME
    parser.add_argument('--np', default=False, type=lambda x: (str(x).lower() == 'true'))  # Optional argument to save image as np array
    parser.add_argument('--t', default=False, type=lambda x: (str(x).lower() == 'true'))  # Optional argument to threshold image

    parser.add_argument('--ft', '--ft', type=int, help='find threshold')  # Optional argument to find threshold image
    parser.add_argument('--multi', default=False, type=lambda x: (str(x).lower() == 'true'))  # Argument for using multiprocessing
    parser.add_argument('--std', '--std', type=int, help='number of standard deviations to threshold with')
    args = parser.parse_args()  # Parse arguments
    print("End")
    print("self.save_jpg " + str(args.jpg))
    print("self.save_np " + str(args.np))
    print("self.threshold " + str(args.t))
    print("self.num_images " + str(args.num))
    print("self.find_threshold_bool " + str(args.ft))
    print("self.multi " + str(args.multi))
    print("self.std " + str(args.std))
    process = Process(args.jpg, args.np, args.t, args.num, args.ft,
                      args.multi, args.std)  # Create new process instance
    system = PySpin.System.GetInstance()
        # Get camera list
    cam_list = system.GetCameras()
    if len(cam_list) == 0:
        print("No cameras found.")
    cam = cam_list.GetByIndex(0)
    cam.Init()
    process.whole_capture(cam)
    cam.EndAcquisition()
    cam.DeInit()
    del cam
    cam_list.Clear()
    del cam_list
    system.ReleaseInstance()
