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
    cam = None
    std = -1

    def __init__(self, save_jpg, save_np, threshold, num_images, find_threshold_bool, multi, std):
        self.save_jpg = save_jpg
        self.save_np = save_np
        self.threshold = threshold
        self.num_images = num_images
        self.find_threshold_bool = find_threshold_bool
        self.multi = multi
        self.std = std
        print("In CMOS")
        print(self.save_jpg)
        print(self.save_np)
        print(self.threshold)
        print(self.num_images)
        print(self.find_threshold_bool)
        print(self.multi)
        print(self.std)

    def find_threshold(self, image, num_images):
        t0 = time.time()
        width = image.shape[1]
        height = image.shape[0]
        img_dev = np.empty((height, width))
        NUM_IMAGES = num_images
        images = np.empty((NUM_IMAGES, height, width), dtype="uint8")
        for i in range(0, NUM_IMAGES):
            temp_img = self.cam.GetNextImage()
            temp_img = temp_img.GetNDArray()
            images[i] = temp_img
            print("Adding image " + str(i) + "...")
        try:
            print("Calculating variance...")
            y = np.var(images, axis=0, dtype=np.dtype("uint16"))
            print("Calculating square roots...")
            y = np.sqrt(y, dtype="float32")
            print("Calculating standard deviations...")
            img_dev = np.mean(images, axis=0) + (y * self.std)
        except KeyboardInterrupt:
            print("Program terminated.")
        t1 = time.time()
        cv2.imwrite("Images/Threshold.jpg", img_dev)
        return img_dev

    def convert_images(self, temp_image_to_be_thresholded, temp_threshold, pic_num):
        temp_image_to_be_thresholded[temp_image_to_be_thresholded > temp_threshold] = 255
        temp_image_to_be_thresholded[temp_image_to_be_thresholded <= temp_threshold] = 0
        if self.save_jpg and self.threshold:  # If user wants to save image as .jpg, save as .jpg
            cv2.imwrite(self.save_folder + 'Processed Picture' + str(pic_num) + '.jpg', temp_image_to_be_thresholded)
        if self.save_np and self.threshold:  # If user wants to save image as .npp, save as .npp
            np.save(self.save_folder + "Processed Array " + str(pic_num), temp_image_to_be_thresholded)

    def setup(self):
        self.system = PySpin.System.GetInstance()

        # Get camera list
        cam_list = self.system.GetCameras()
        if len(cam_list) == 0:
            print("No cameras found.")
            return None
        self.cam = cam_list.GetByIndex(0)
        self.cam.Init()
        exposure_time = 200  # in milliseconds
        capture_fps = 4.9
        gain = 0  # ISO for digital cameras
        reverse_x = False
        reverse_y = False
        bit = 8
        # Get PySpin system
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)
        print("Camera firmware version: " + self.cam.DeviceFirmwareVersion.ToString())
        # load default config
        self.cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
        self.cam.UserSetLoad()
        # set acquisition to continuous, turn off auto exposure, set the frame rate
        # Camera Settings
        # Set Packet Size
        self.cam.GevSCPSPacketSize.SetValue(9000)
        self.cam.DeviceLinkThroughputLimit.SetValue(125000000)
        # Set acquistion mode
        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        # Set exposure time
        self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        self.cam.ExposureMode.SetValue(PySpin.ExposureMode_Timed)
        self.cam.ExposureTime.SetValue(exposure_time * 1e3)
        # Set FPS
        self.cam.AcquisitionFrameRateEnable.SetValue(True)
        self.cam.AcquisitionFrameRate.SetValue(capture_fps)
        # set analog, gain, turn off gamma
        self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        self.cam.Gain.SetValue(gain)
        self.cam.GammaEnable.SetValue(False)
        self.cam.ReverseX.SetValue(reverse_x)
        self.cam.ReverseY.SetValue(reverse_y)
        if bit > 8:  # If requested bit level is above 8, make it 16
            image_bit = 16
            self.cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit12)
            self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono12p)
        else:  # Otherwise, make it 8
            image_bit = 8
            self.cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit10)
            self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
        self.cam.BeginAcquisition()

    def whole_capture(self):
        self.setup()
        processes = []
        threshold_img = None
        i = 0
        try:
            while True:  # Take n images
                if i == self.num_images:
                    break
                if i == 0:
                    image = self.cam.GetNextImage()
                    image_np = image.GetNDArray()
                    if self.find_threshold_bool != -1:  # If the user needs to find a threshold image,
                        threshold_img = self.find_threshold(image_np,
                                                            self.find_threshold_bool)  # Find a threshold image
                        self.find_threshold_bool = -1  # Make sure it only goes through this process once as it is expensive
                    else:
                        threshold_img = cv2.imread(
                            "Images/Threshold.jpg")  # If they don't want to find a new threshold, pull the old one
                        threshold_img = threshold_img[:, :, 0]
                print("Capturing image " + str(i))
                image = self.cam.GetNextImage()
                if image.IsIncomplete():
                    print("Error: image is incomplete.")
                    continue
                image_np = np.copy(image.GetNDArray())  # Convert image to nd array
                if self.threshold:  # If they want thresholding,
                    if self.multi:  # If they want multiprocessing,
                        multiprocessing_threshold_image = multiprocessing.Process(target=self.convert_images, args=(
                            image_np, threshold_img, i,))  # Create a call to multiprocessing
                        processes.append(multiprocessing_threshold_image)  # Append it to list of processes
                        multiprocessing_threshold_image.start()  # Start process
                    else:
                        self.convert_images(image_np, threshold_img, i)  # Otherwise, process in standard format
                if self.save_jpg and self.threshold:  # If user wants to save image as .jpg, save as .jpg
                    cv2.imwrite(self.save_folder + 'Unprocessed Picture' + str(i) + '.jpg',
                                image)
                if self.save_np and self.threshold:  # If user wants to save image as .npp, save as .npp
                    np.save(self.save_folder + "Unprocessed Array " + str(i), image_np)
                del image
                image_np = None
                i += 1
        except KeyboardInterrupt:  # If keyboard interrupt (ctrl+c) is found, kill loop and print message
            print('Process interrupted.')

        if self.multi:  # If multiprocessing was used, clean up
            for process_temp in processes:
                process_temp.join()

        self.cam.EndAcquisition()
        self.cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
        self.cam.UserSetLoad()
        self.cam.DeInit()
        del self.cam
        self.system.ReleaseInstance()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parses arguments")  # Create argument parser
    parser.add_argument('--num', type=int,
                        help='number of images to collect')  # Positional argument to specify number of images
    parser.add_argument('--jpg', '--jpg', help='save image as jpg')  # Optional argument to save images as .jpg
    parser.add_argument('--np', '--np', help='save image as np array')  # Optional argument to save image as np array
    parser.add_argument('--t', '--threshold', help="threshold image")  # Optional argument to threshold image

    parser.add_argument('--ft', '--ft', type=int, help='find threshold')  # Optional argument to find threshold image
    parser.add_argument('--multi', '--multi',
                        help='Determines whether multiprocessing should be used')  # Argument for using multiprocessing
    parser.add_argument('--std', '--std', type=int, help='number of standard deviations to threshold with')
    args = parser.parse_args()  # Parse arguments
    process = Process(args.jpg, args.np, args.t, args.num, args.ft,
                      args.multi, args.std)  # Create new process instance
    process.whole_capture()
