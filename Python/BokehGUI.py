import numpy as np
import cv2
from bokeh.layouts import column, row
from bokeh.plotting import figure, output_file, curdoc
from bokeh.models.widgets import Toggle, Slider
from bokeh.transform import linear_cmap
import os

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

global_images = []
global_image = np.zeros((3, 3))

class Process():
    save_jpg = False  # Instance fields to determine what "path" code will take, for example:
    # Should images be saved or not?
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
        width = image_threshold.shape[1]  # Find dimensions of the image
        height = image_threshold.shape[0]
        img_dev = np.empty((height, width))  # Create an empty array to hold the final threshold image
        NUM_IMAGES = num_images
        # Create an empty array that will
        # Contain all of the images that will be used for finding standard deviations
        # It should be deep enough to hold all images, with the height and width at the dimensions of the image
        images_threshold = np.empty((NUM_IMAGES, height, width), dtype="float64")
        for i in range(0, NUM_IMAGES):  # Take the number of pictures to be used for finding standard deviations
            temp_img = cam.GetNextImage()  # Take the picture
            temp_img = temp_img.GetNDArray()  # Get a numpy array from it
            images_threshold[i] = temp_img  # Add the numpy array to the array of images
        try:
            print("Thresholding...")
            y = np.var(images_threshold, axis=0, dtype=np.dtype("float64"))  # Find variance of the images
            y = np.sqrt(y, dtype="float64")  # Find square root of the variance (faster than np.std())
            # Add the mean pixel value to the standard deviation multiplied by the sigma multiple
            img_dev = np.mean(images_threshold, axis=0) + (y * self.std)
        except KeyboardInterrupt:
            print("Program terminated.")
        np.clip(img_dev, 0, 255)  # Make sure all values are between 0 and 255 ( Range of pixel values )
        cv2.imwrite("Images/Threshold.tiff", img_dev)  # Save image to file
        print("Done thresholding")
        # Return the image, which is the mean image of all images taken plus the number of std deviations
        return img_dev

    def convert_images(self, temp_image_to_be_thresholded, temp_threshold, pic_num):
        # Replace all values greater than the threshold with a purely white pixel
        temp_image_to_be_thresholded[temp_image_to_be_thresholded > temp_threshold] = 255
        # Replace all values less than the threshold with a purely black pixel
        temp_image_to_be_thresholded[temp_image_to_be_thresholded <= temp_threshold] = 0
        global_images.append(temp_image_to_be_thresholded)
        if self.save_jpg:  # If user wants to save image as .tiff, save as .tiff
            cv2.imwrite(self.save_folder + 'Processed Picture' + str(pic_num) + '.tiff', temp_image_to_be_thresholded)
        if self.save_np:  # If user wants to save image as .npp, save as .npp
            np.save(self.save_folder + "Processed Array " + str(pic_num), temp_image_to_be_thresholded)

    def setup(self, cam):
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
        self.setup(cam)
        processes = []
        threshold_img = None
        i = 0
        try:

            while True:  # Take n images
                # If we have taken the desired number of images, break loop
                if i == self.num_images:
                    break
                if i == 0:  # If it's the first image being taken,
                    image_first = cam.GetNextImage()  # Take a sample photo to be used for the find_threshold() method
                    image_np_first = image_first.GetNDArray().astype("uint32")
                if self.find_threshold_bool != -1:  # If the user needs to find a threshold image,
                    threshold_img = self.find_threshold(image_np_first,
                                                        self.find_threshold_bool)  # Find a threshold image
                    self.find_threshold_bool = -1  # Make sure it only goes through this process once as it is expensive
                else:
                    threshold_img = cv2.imread(
                        "Images/Threshold.tiff")  # If they don't want to find a new threshold, pull the old one
                    threshold_img = threshold_img[:, :, 0]  # Make sure dimensions are (width, height, 1)
                print("Capturing image " + str(i))
                image = cam.GetNextImage()  # Take picture
                if image.IsIncomplete():  # If its incomplete, try again
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


def main():
    parser = argparse.ArgumentParser(description="Parses arguments")  # Create argument parser
    parser.add_argument('--num', type=int,
                        help='number of images to collect')  # Positional argument to specify number of images
    parser.add_argument('--jpg', default=False, type=lambda x: (str(x).lower() == 'true'))  # FIXME
    parser.add_argument('--np', default=False,
                        type=lambda x: (str(x).lower() == 'true'))  # Optional argument to save image as np array
    parser.add_argument('--t', default=False,
                        type=lambda x: (str(x).lower() == 'true'))  # Optional argument to threshold image

    parser.add_argument('--ft', '--ft', type=int, help='find threshold')  # Optional argument to find threshold image
    parser.add_argument('--multi', default=False,
                        type=lambda x: (str(x).lower() == 'true'))  # Argument for using multiprocessing
    parser.add_argument('--std', '--std', type=int, help='number of standard deviations to threshold with')
    args = parser.parse_args()  # Parse arguments
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


#######################################################################################################################
#   Separation of GUI and Functionality | Separation of GUI and Functionality | Separation of GUI and Functionality   #
#######################################################################################################################


# Changes value of settings['save_jpg']
def save_jpg_handler(attr, old, new):
    settings['save_jpg'] = not settings['save_jpg']  # Change value to opposite of what it currently is


# Changes value of settings['threshold']
def threshold_handler(attr, old, new):
    settings['threshold'] = not settings['threshold']  # Change value to opposite of what it currently is


# Changes value of settings['save_np'}
def save_np_handler(attr, old, new):
    settings['save_np'] = not settings['save_np']  # Change value to opposite of what it currently is


# Changes value of settings['find_threshold_num'}
def find_threshold_num_handler(attr, old, new):
    settings['find_threshold_num'] = find_threshold_num_slider.value  # Change value to what the slider reads


# Changes the value of settings['multi']
def multi_handler(attr, old, new):
    settings['multi'] = not settings['multi']  # Change value to opposite of what it currently is


# Changes the value of settings['num_images']
def num_images_handler(attr, old, new):
    settings['num_images'] = num_images_slider.value  # Change the value to what the slider reads


# Runs program when button is clicked
def run_button_handler(attr, old, new):
    # Run python script from command line
    animate()

def animate():
    global callback_id
    callback_id = curdoc().add_periodic_callback(plot_update, 200)


def plot_update(attrname, old, new):
    global_image = global_images[len(global_images) - 1]

def std_handler(attr, old, new):
    settings['std'] = std_slider.value
    print(str(settings['std']))


settings = {
    'save_jpg': False,  # Determines whether files should be saved as jpgs
    'threshold': False,  # If images should be thresholded
    'save_np': False,  # If images should be saved as np arrays
    'find_threshold_num': -1,  # How many images (if any) should be used to find new threshold image
    'multi': False,  # If multiprocessing should be used
    'num_images': -1,  # How many images should be taken
    'std': 3  # How many standard deviations should be used for
}

# Create toggles and sliders to determine the values of the elements of settings{}
save_jpg_toggle = Toggle(active=False, label="Do you want to save images as .jpgs?", button_type='success')
save_np_toggle = Toggle(active=False, label='Do you want to save images as numpy arrays?', button_type="success")
threshold_toggle = Toggle(active=False, label="Do you want to threshold images?", button_type="success")
find_threshold_num_slider = Slider(start=-1, end=300, value=-1, title="Do you want to find a new picture to be used"
                                                                      " for thresholding? If so, how many images would"
                                                                      " you like it to be composed of? Otherwise, "
                                                                      "leave at -1.")
multi_toggle = Toggle(active=False, label='Do you want to use multiprocessing?', button_type="success")
num_images_slider = Slider(start=-1, end=300, value=-1, title='How many images would you like to take? Leave at -1 for '
                                                              'images to be taken until manually stopped.')
run_button = Toggle(active=False, label="Start collecting images", button_type="primary")
std_slider = Slider(start=0, end=20, value=3, title="How many standard deviations would you like to threshold with?")
which_image_slider = Slider(start=0, end=100000000, value=0, title="Image: ")
# prepare some data
img = cv2.imread("image.jpg")

# output to static HTML file (with CDN resources)
output_file("CMOS_Sensor.html", title="BokehGUI.py example", mode="cdn")

TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"

# create a new plot with the tools above, and explicit ranges
p = figure(tools=TOOLS, x_range=(0, img.shape[1]), y_range=(0, img.shape[0]))
# add a circle renderer with vectorized colors and sizes
p.circle(np.where(global_image > 0)[0], np.where(global_image > 0)[1], radius=1,
         fill_color=linear_cmap('counts', 'Viridis'), fill_alpha=0.6)

# Make it so the values in the python script change when they change in the GUI
save_jpg_toggle.on_change("active", save_jpg_handler)
save_np_toggle.on_change("active", save_np_handler)
threshold_toggle.on_change("active", threshold_handler)
multi_toggle.on_change("active", multi_handler)
find_threshold_num_slider.on_change("value", find_threshold_num_handler)
run_button.on_change("active", run_button_handler)
num_images_slider.on_change("value", num_images_handler)
std_slider.on_change("value", std_handler)
which_image_slider.on_change('value', plot_update)

# show the GUI in browser
curdoc().add_root(
    row(column(save_np_toggle, save_jpg_toggle, threshold_toggle, multi_toggle, find_threshold_num_slider, std_slider,
               num_images_slider, run_button, p)))  # FIXME p will eventually show images, currently not
curdoc().title = "CMOS Process"
