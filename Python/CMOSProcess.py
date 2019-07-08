import numpy as np
import argparse
import datetime
#import PySpin
import cv2
import time
import multiprocessing

"""CMOSProcess.py: Pulls images from FLIR CMOS Cameras and processes them for Plasma Spectroscopy."""

__author__ = "Hunter Abraham"
__credits__ = ["Adam Schoenwald"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Hunter Abraham"
__email__ = "hjabraham@wisc.edu"
__status__ = "Production"


class Process():
    save_jpg = False  # Static methods to determine what "path" code will take
    save_np = False
    threshold = False
    find_threshold_bool = False
    multi = False
    num_images = -1
    save_folder = ""
    cam = None

    def __init__(self, save_jpg, save_np, threshold, num_images, save_folder, find_threshold_bool, multi):
        if save_jpg is not None:
            self.save_jpg = save_jpg
        if save_np is not None:
            self.save_np = save_np
        if threshold is not None:
            self.threshold = threshold
        if num_images != -1:
            self.num_images = num_images
        if save_folder is not None:
            self.save_folder = save_folder
        if find_threshold_bool is not None:
            self.find_threshold_bool = find_threshold_bool
        if multi is not None:
            self.multi = multi

    def get_threshold(self):
        return self.threshold

    def set_threshold(self, threshold):
        self.threshold = threshold

    def find_threshold(self, image):
        t0 = time.time()
        save_folder = "C:\\Users\\habraha1\\Documents\\Images\\"
        print(cv2.__version__)
        print(image.shape)
        width = image.shape[1]
        height = image.shape[0]
        sigma_multiple = 3
        img_dev = np.empty((height, width))
        NUM_IMAGES = 200
        images = np.empty((NUM_IMAGES, height, width), dtype="uint8")
        for i in range(0, NUM_IMAGES):
            temp_img = cam.GetNextImage()
            temp_img = temp_img.GetNDArray()
            temp_img = temp_img[:, :, 0]
            images[i] = temp_img
            print("Adding image " + str(i) + "...")
        print("Images shape: " + str(images.shape))
        try:
            print("Calculating variance...")
            y = np.var(images, axis=0, dtype=np.dtype("uint16"))
            print("Calculating square roots...")
            y = np.sqrt(y, dtype="float32")
            print("Calculating standard deviations...")
            img_dev = np.mean(images, axis=0) + (y * sigma_multiple)

        except KeyboardInterrupt:
            print("Program terminated.")

        t1 = time.time()

        print("Total time to run: " + str(t1 - t0))
        print(img_dev)
        print(img_dev.shape)
        return img_dev

    def convert_images(self, threshold_image, temp_threshold):
        threshold_image[threshold_image > temp_threshold] = 255
        threshold_image[threshold_image <= temp_threshold] = 0

    def capture_images(self, cam, num_images):
        images = []
        for i in range(0, num_images):
            image = cam.GetNextImage()
            images.append(image)
        return images

    def setup(self):
        exposure_time = 200  # in milliseconds
        capture_fps = 4.9
        gain = 0  # ISO for digital cameras
        reverse_x = False
        reverse_y = False
        bit = 8
        # Get PySpin system
        system = PySpin.System.GetInstance()
        # Get camera list
        cam_list = system.GetCameras()
        self.cam = cam_list.GetByIndex(0)
        self.cam.Init()
        print("Camera firmware version: " + self.cam.DeviceFirmwareVersion.ToString())
        # load default config
        self.cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
        self.cam.UserSetLoad()
        # set acquisition to continuous, turn off auto exposure, set the frame rate
        # Camera Settings
        # Set Packet Size
        self.cam.GevSCPSPacketSize.SetValue(9000)
        self.cam.DeviceLinkThroughputLimit.SetValue(100000000)
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

    def save_images_jpg(self, images):
        i = 0
        for temp_img in images:
            cv2.imwrite("Image " + str(i) + ".jpg", temp_img)
            i += 1

    def save_images_np(self, images):
        n = 0
        for np_img in images:
            np.save("NpArray " + str(datetime.time()), np_img)
            n += 1

    def whole_capture(self):
        self.setup()
        processes = []
        if self.num_images == -1:  # If number of images is not specified,
            try:  # Continue until the program is interrupted
                while True:
                    image = self.cam.GetNextImage()  # Capture image
                    image_np = image.GetNDArray()
                    if self.save_jpg:  # If save as jpg is specified,
                        cv2.imwrite(self.save_folder + "Picture " + str(time.time()) + ".jpg",
                                    image_np)  # Save np as jpg
                    if self.save_np:  # Otherwise,
                        np.save(self.save_folder + "Picture " + str(time.time()), image_np)  # Save as np array
                    if self.find_threshold_bool:  # If the user wants to find the threshold, call method
                        threshold_img = self.find_threshold(self.cam)
                        self.find_threshold_bool = False  # Make sure that the threshold is found only once
                    else:
                        threshold_img = cv2.imread(
                            self.save_folder + "Threshold.jpg")  # If the user doesn't want to create a threshold image, find one
                    if self.threshold:
                        if self.multi:  # If user wants to use multiprocessing, call method with multiprocessing
                            multiprocessing_convert_image = multiprocessing.Process(target=threshold_img,
                                                                                    args=(image_np,))
                            # append to list of processes
                            processes.append(multiprocessing_convert_image)
                            # start
                            multiprocessing_convert_image.start()
                        else:  # Otherwise, use single core
                            self.threshold(image, threshold_img)
            except KeyboardInterrupt:  # If keyboard interrupt (ctrl+c) is found, kill loop and print message
                print('Process interrupted.')

            if self.multi:  # Clean up multiprocessing if it was used
                for process_temp in processes:
                    process_temp.join()

        else:  # If a number of images was specified,
            processes = []
            for i in range(0, self.num_images):  # Take n images
                image = self.cam.GetNextImage()
                image_np = image.GetNDArray()  # Convert image to nd array
                if self.save_jpg:  # If user wants to save image as .jpg, save as .jpg
                    cv2.imwrite(self.save_folder + "Picture " + str(time.time()) + ".jpg", image_np)
                if self.save_np:  # If user wants to save image as .npp, save as .npp
                    np.save(self.save_folder + "Picture " + str(time.time()) + ".jpg", image_np)
                if self.find_threshold_bool:  # If the user needs to find a threshold image,
                    threshold_img = self.find_threshold(image_np)  # Find a threshold image
                    self.find_threshold_bool = False  # Make sure it only goes through this process once as it is expensive
                else:
                    threshold_img = cv2.imread(
                        self.save_folder + "Threshold.jpg")  # If they don't want to find a new threshold, pull the old one
                if self.threshold:  # If they want thresholding,
                    if self.multi:  # If they want multiprocessing,
                        multiprocessing_threshold_image = multiprocessing.Process(target=threshold_img, args=(
                        image_np,))  # Create a call to multiprocessing
                        processes.append(multiprocessing_threshold_image)  # Append it to list of processes
                        multiprocessing_threshold_image.start()  # Start process
                    else:
                        self.threshold(image, threshold_img)  # Otherwise, process in standard format
            if self.multi:  # If multiprocessing was used, clean up
                for process_temp in processes:
                    process_temp.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parses arguments")  # Create argument parser
    parser.add_argument('path', metavar='P', type=str, nargs='1',
                        help='path to save file')  # Create positional argument for save path
    parser.add_argument('--jpg', '--jpg', help='save image as jpg')  # Optional argument to save images as .jpg
    parser.add_argument('--np', '--np', help='save image as np array')  # Optional argument to save image as np array
    parser.add_argument('--t', '--threshold', help="threshold image")  # Optional argument to threshold image
    parser.add_argument('num', metavar='N', type=int, nargs=1,
                        help='number of images to collect')  # Positional argument to specify number of images
    parser.add_argument('--ft', '--ft', help='find threshold')  # Optional argument to find threshold image
    parser.add_argument('--multi', '--multi',
                        help='Determines whether multiprocessing should be used')  # Argument for using multiprocessing
    args = parser.parse_args()  # Parse arguments
    process = Process(args.jpg, args.np, args.threshold, args.num, args.path, args.ft,
                      args.multi)  # Create new process instance
    process.whole_capture()  # Begin to capture images
