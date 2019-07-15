import numpy as np
import cv2
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.plotting import figure, output_file, show
from bokeh.models.widgets import TextInput, Toggle, Slider, Button
from bokeh.models.sources import ColumnDataSource
import os
import pandas as pd
import CMOSProcess


def save_jpg_handler(attr, old, new):
    settings['save_jpg'] = not settings['save_jpg']
    print(settings['save_jpg'])


def threshold_handler(attr, old, new):
    settings['threshold'] = not settings['threshold']
    print(settings['threshold'])


def save_np_handler(attr, old, new):
    settings['save_np'] = not settings['save_np']
    print(settings['save_np'])


def find_threshold_bool_handler(attr, old, new):
    settings['find_threshold_bool'] = find_threshold_bool_slider.value
    print(settings['find_threshold_bool'])


def multi_handler(attr, old, new):
    settings['multi'] = not settings['multi']
    print(settings['multi'])


def num_images_handler(attr, old, new):
    settings['num_images'] = num_images_slider.value
    print(settings['num_images'])


def run_button_handler(attr, old, new):
    os.system(
        "python3 CMOSProcess.py --jpg=" + str(settings['save_jpg']) + " --np=" + str(settings['save_np']) + " --t=" +
        str(settings['threshold']) + " --ft=" + str(settings['find_threshold_bool']) + " --multi="
        + str(settings['multi']) + " --num=" + str(settings['num_images']) + " --std=" + str(settings['std']) + " &")


def std_handler(attr, old, new):
    settings['std'] = std_slider.value
    print(str(settings['std']))


def animate_update():
    slider.value += 1
    global source
    global img
    #image_data = cv2.imread("Processed Picture " + slider.value + ".tiff")
    image_data = cv2.imread("Images/Processed Picture" + str(slider.value) + ".tiff")
    image_data = image_data[:,:,0]
    np.delete(image_data, list(range(0, image_data.shape[1], 2)), axis=0) # Delete every other row to speed up image load
    np.delete(image_data, list(range(0, image_data.shape[1], 2)), axis=1)
    image_data = np.flipud(image_data)
    source.data = dict(image=[image_data])


def animate():
    global callback_id
    if button.label == 'Start streaming images':
        button.label = 'Stop streaming images'
        callback_id = curdoc().add_periodic_callback(animate_update, 1000)
    else:
        button.label = 'Start streaming images'
        curdoc().remove_periodic_callback(callback_id)


slider = Slider(start=0, end=1000, value=0, step=1, title="Picture (Do not touch)")
button = Button(label='Start streaming images', width=60)
button.on_click(animate)
settings = {
    'save_jpg': False,
    'threshold': False,
    'save_np': False,
    'find_threshold_bool': -1,
    'multi': False,
    'num_images': -1,
    'std': 3
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
std_slider = Slider(start=0, end=20, value=3, title="How many standard deviations would you like to threshold with?")
# prepare some data

img = cv2.imread("Images/Threshold.tiff")
img = img[:, :, 0]
img = np.flipud(img)
images = [img]
x_vals = np.arange(0, img.shape[0])
# output to static HTML file (with CDN resources)
output_file("CMOS_Sensor.html", title="BokehGUI.py example", mode="cdn")

TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"
source = ColumnDataSource(data=dict(image=[img]))

# create a new plot with the tools above, and explicit ranges
p = figure(tools=TOOLS, x_range=(0, img.shape[1]), y_range=(0, img.shape[0]))
# add a circle renderer with vectorized colors and sizes
p.image(image='image', source=source, x=0, y=0, dw=img.shape[1], dh=img.shape[0], palette='Greys256')

save_jpg_toggle.on_change("active", save_jpg_handler)
save_np_toggle.on_change("active", save_np_handler)
threshold_toggle.on_change("active", threshold_handler)
multi_toggle.on_change("active", multi_handler)
find_threshold_bool_slider.on_change("value", find_threshold_bool_handler)
run_button.on_change("active", run_button_handler)
num_images_slider.on_change("value", num_images_handler)
std_slider.on_change("value", std_handler)
# show the results
curdoc().add_root(
    row(column(save_np_toggle, save_jpg_toggle, threshold_toggle, multi_toggle, find_threshold_bool_slider, std_slider,
               num_images_slider, run_button, p, slider, button)))
