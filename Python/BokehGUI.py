import numpy as np
import cv2
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Button
from bokeh.plotting import figure, curdoc
from bokeh.plotting import figure, output_file, show
from bokeh.models.widgets import TextInput, Toggle, Slider
import os


def save_jpg_handler(attr, old, new):
    print("Previous label: " + old)
    print("Updated label: " + new)


def threshold_handler(attr, old, new):
    settings['threshold'] = not settings['threshold']


def save_np_handler(attr, old, new):
    settings['save_np'] = not settings['save_np']


def find_threshold_bool_handler(attr, old, new):
    settings['find_threshold_bool'] = find_threshold_bool_slider.value


def multi_handler(attr, old, new):
    settings['multi'] = not settings['multi']


def num_images_handler(attr, old, new):
    settings['num_images'] = num_images_slider.value


def run_button_handler(attr, old, new):
    os.system("python3 ImageProcess.py --jpg=" + str(settings['save_jpg']) + " --np=" + str(settings['save_np']) + " --t=" +
              str(settings['threshold']) + " --ft=" + str(settings['find_threshold_bool']) + " --multi="
              + str(settings['multi']))




settings = {
    'save_jpg': False,
    'threshold': False,
    'save_np': False,
    'find_threshold_bool': -1,
    'multi': False,
    'num_images': -1
}

save_jpg_toggle = Toggle(active=False, label="Do you want to save images as .jpgs?", button_type='success')
save_np_toggle = Toggle(active=False, label='Do you want to save images as numpy arrays?', button_type="success")
threshold_toggle = Toggle(active=False, label="Do you want to threshold images?", button_type="success")
find_threshold_bool_slider = Slider(start=-1, end=300, value=-1, title="Do you want to find a new picture to be used"
                                                                       " for thresholding? If so, how many images would"
                                                                       " you like it to be composed of? Otherwise, "
                                                                       "leave at -1.")
multi_toggle = Toggle(active=False, label='Do you want to use multiprocessing?', button_type="success")
num_images_slider = Slider(start=-1, end=300, value=-1, title='How many images would you like to take? Leave at -1 for '
                                                              'images to be taken until manually stopped.')
run_button = Toggle(active=False, label="Start collecting images", button_type="primary")
# prepare some data
img = cv2.imread("image.jpg")

# output to static HTML file (with CDN resources)
output_file("CMOS_Sensor.html", title="BokehGUI.py example", mode="cdn")

TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"

# create a new plot with the tools above, and explicit ranges
p = figure(tools=TOOLS, x_range=(0, img.shape[1]), y_range=(0, img.shape[0]))

# add a circle renderer with vectorized colors and sizes
p.circle(img.shape[1], img.shape[0], radius=1, fill_color='green', fill_alpha=0.6)
save_jpg_toggle.on_change("active", save_jpg_handler)
save_np_toggle.on_change("active", save_np_handler)
threshold_toggle.on_change("active", threshold_handler)
multi_toggle.on_change("active", multi_handler)
find_threshold_bool_slider.on_change("value", find_threshold_bool_handler)
run_button.on_change("active", run_button_handler)
# show the results
curdoc().add_root(
    row(column(save_np_toggle, save_jpg_toggle, threshold_toggle, multi_toggle, find_threshold_bool_slider,
               num_images_slider, run_button), column(p)))

#input("Press any button when you are ready to process")

