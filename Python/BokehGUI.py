import numpy as np
import cv2
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.plotting import figure, output_file, show
from bokeh.models.widgets import TextInput, Toggle, Slider, Button
from bokeh.models.sources import ColumnDataSource
import os
import json
import pandas as pd


def save_jpg_handler(attr, old, new):
    settings_json['save_jpg'] = not settings_json['save_jpg']  # Flip value of save_jpg
    if save_jpg_toggle.button_type == 'danger':  # Change the button color from red to green or green to red
        save_jpg_toggle.button_type = 'success'
    else:
        save_jpg_toggle.button_type = 'danger'


def threshold_handler(attr, old, new):
    settings_json['threshold'] = not settings_json['threshold']  # Flip the value of threshold
    if threshold_toggle.button_type == 'danger':  # Change the button color from red to green or green to red
        threshold_toggle.button_type = 'success'
    else:
        threshold_toggle.button_type = 'danger'


def save_np_handler(attr, old, new):
    settings_json['save_np'] = not settings_json['save_np']  # Flip the value of save_np
    if save_np_toggle.button_type == 'success':  # Change the button color from red to green or green to red
        save_np_toggle.button_type = 'danger'
    else:
        save_np_toggle.button_type = 'success'


def find_threshold_bool_handler(attr, old, new):
    settings_json[
        'find_threshold_bool'] = find_threshold_bool_slider.value  # Set the value of the setting to the slider value


def multi_handler(attr, old, new):
    settings_json['multi'] = not settings_json['multi']  # Flip the value of multi
    if multi_toggle.button_type == 'success':  # Change the color of the button from green to red or red to green
        multi_toggle.button_type = 'danger'
    else:
        multi_toggle.button_type = 'success'


def num_images_handler(attr, old, new):
    settings_json['num_images'] = num_images_slider.value  # Set the value of num_images to the selected slider value


def run_button_handler(attr, old, new):
    if run_button.button_type == 'primary':  # Change the run button color from blue to red or red to blue
        run_button.button_type = 'danger'
    else:
        run_button.button_type = 'primary'

    os.system(
        "python3 CMOSProcess.py --jpg=" + str(settings_json['save_jpg']) + " --np=" + str(settings_json['save_np']) + " --t=" +
        str(settings_json['threshold']) + " --ft=" + str(settings_json['find_threshold_bool']) + " --multi="
        + str(settings_json['multi']) + " --num=" + str(settings_json['num_images']) + " --std=" + str(settings_json['std'])
        + " --address=" + str(settings_json['packet_destination']) + " --exposure=" + str(settings_json['exposure']) + " --gain=" + str(settings_json['gain'])
        + " &")  # run command in OS


def std_handler(attr, old, new):
    settings_json['std'] = std_slider.value  # Change value of setting to selected slider value


def textbox_handler(attr, old, new):
    settings_json['packet_destination'] = packet_destination_textbox.value


def animate_update():
    slider.value += 1  # Increase slider value
    global source  # "Import" global values
    global img
    image_data = cv2.imread("Images/Processed Picture " + str(slider.value) + ".tiff")  # Read in next image
    image_data = image_data[:, :, 0]  # Convert to mono
    image_data = np.flipud(image_data)  # Flip right-side up
    source.data = dict(image=[image_data])  # Update source


def animate():
    global callback_id  # "Import" global values
    if button.label == 'Start streaming images':  # If start is clicked, change button value and start periodic call
        button.label = 'Stop streaming images'
        callback_id = curdoc().add_periodic_callback(animate_update, 1000)
    else:
        button.label = 'Start streaming images'  # If stop button is clicked, change button label and stop periodic call
        curdoc().remove_periodic_callback(callback_id)


def science_button_handler():
    # Turn on thresholding
    threshold_toggle.button_type = 'success'
    settings_json['threshold'] = True
    # Turn on multiprocessing
    multi_toggle.button_type = 'success'
    num_images_slider.value = -1
    find_threshold_bool_slider.value = -1
    settings_json['multi'] = True
    save_jpg_toggle.button_type = 'danger'
    settings_json['save_jpg'] = False
    save_np_toggle.button_type = 'danger'
    settings_json['save_np'] = False
    settings_json['find_threshold_bool'] = -1
    find_threshold_bool_slider.value = -1
    settings_json['num_images'] = -1
    num_images_slider.value = -1
    settings_json['std'] = 3
    std_slider.value = -1
    settings_json['packet_destination'] = packet_destination_textbox.value

def exposure_handler(attr, old, new):
    settings_json['exposure'] = int(exposure_textbox.value)
    print(settings_json)

def gain_handler(attr, old, new):
    settings_json['gain'] = float(gain_textbox.value)


def threshold_button_handler():
    # save_jpg ON
    settings_json['save_jpg'] = True
    save_jpg_toggle.button_type = 'success'
    # Find threshold value
    find_threshold_bool_slider.value = 10
    settings_json['find_threshold_bool'] = 10
    num_images_slider.value = 1
    settings_json['num_images'] = 1
    save_np_toggle.button_type = 'danger'
    settings_json['save_np'] = False,
    settings_json['std'] = 3
    settings_json['packet_destination'] = packet_destination_textbox.value

def save_settings():
    json_f = json.dumps(settings_json) # dump dict as string
    f = open("/home/schoen/imageprocess.json", "w") # open json that holds settings
    f.write(json_f) # write new settings to file
    f.close() # close file


def load_settings():
    global settings_json
    global json_file
    # open file
    f = open("/home/schoen/imageprocess.json")
    # Turn it into a dictionary
    settings_json = dict(json.load(f))

    # Go through each setting and adjust both the value in the dictionary and the way it is represented in the GUI
    if settings_json['save_jpg'] == True:
        settings_json['save_jpg'] = True
        save_jpg_toggle.button_type='success'
    else:
        settings_json['save_jpg'] = False
        save_jpg_toggle.button_type='danger'

    if settings_json['threshold'] == True:
        settings_json['threshold'] = True
        threshold_toggle.button_type = 'success'
    else:
        settings_json['threshold'] = False
        threshold_toggle.button_type = 'danger'

    if settings_json['save_np'] == True:
        settings_json['save_np'] = True
        save_np_toggle.button_type='success'
    else:
        settings_json['save_np'] = False
        save_np_toggle.button_type='danger'

    if settings_json['multi'] == True:
        settings_json['multi'] = True
        multi_toggle.button_type = 'success'
    else:
        settings_json['multi'] = False
        multi_toggle.button_type = 'danger'

    settings_json['find_threshold_bool'] = int(settings_json['find_threshold_bool'])
    find_threshold_bool_slider.value = int(settings_json['find_threshold_bool'])

    settings_json['num_images'] = int(settings_json['num_images'])
    num_images_slider.value = int(settings_json['num_images'])

    settings_json['std'] = int(settings_json['std'])
    std_slider.value = int(settings_json['std'])
    packet_destination_textbox.value = settings_json['packet_destination']


packet_destination_textbox = TextInput(
    title="What address would you like to send the packet to?")  # Create textinput for data packet address

slider = Slider(start=0, end=1000, value=0, step=1, title="Picture (Do not touch)")  # Create slider to control image #
button = Button(label='Start streaming images', width=60)  # Create button to start streaming images
button.on_click(animate)  # On button click, call animate() method to stream images
json_file = json.load(open('/home/schoen/imageprocess.json'))
settings_json = dict(json_file)
# Create toggle for saving images as .jpgs or .tiffs
if settings_json['save_jpg'] == 'True':
    save_jpg_toggle = Toggle(active=True, label='Do you want to save images as .jpgs?', button_type='success')
    settings_json['save_jpg'] = True
else:
    save_jpg_toggle = Toggle(active=False, label='Do you want to save images as .jpgs?', button_type='danger')
    settings_json['save_jpg'] = False


# Create toggle for saving images as .npy arrays
if settings_json['save_np'] == 'True':
    save_np_toggle = Toggle(active=True, label='Do you want to save images as .npy arrays', button_type='success')
    settings_json['save_np'] = False
else:
    save_np_toggle = Toggle(active=False, label='Do you want to save images as .npy arrays', button_type='danger')
    settings_json['save_np'] = False
# Create toggle for thresholding the images
if settings_json['threshold'] == 'True':
    threshold_toggle = Toggle(active=True, label='Do you want to threshold images?', button_type='success')
    settings_json['threshold'] = True
else:
    threshold_toggle = Toggle(active=False, label='Do you want to threshold images?', button_type='success')
    settings_json['threshold'] = False

# Create slider for finding a new threshold image
find_threshold_bool_slider = Slider(start=-1, end=300, value=int(settings_json['find_threshold_bool']), title='How many'
                                     ' images would you like to use to find new threshold image? Enter -1 to use current'
                                                                                                              'image.')
# Create toggle for using multiprocessing
if settings_json['multi'] == 'True':
    multi_toggle = Toggle(active=True, label='Do you want to use multiprocessing?', button_type='success')
    settings_json['multi'] = True
else:
    multi_toggle = Toggle(active=False, label='Do you want to use multiprocessing?', button_type='danger')
    settings_json['multi'] = False
load_settings_button = Button(label='load settings from imageprocess.json', width=60)
save_settings_button = Button(label='save settings to imageprocess.json', width=60)
# Create slider for the number of images to be captured
num_images_slider = Slider(start=-1, end=300, value=int(settings_json['num_images']), title='How many images would you like to take? Leave at -1 for '
                                                              'images to be taken until manually stopped.')
# Create button to run the program
run_button = Toggle(active=False, label="Start collecting images", button_type="primary")
# Create slider to control the number of standard deviations
std_slider = Slider(start=0, end=20, value=int(settings_json['std']), title="How many standard deviations would you like to threshold with?")
# Create button for "science" mode
science_button = Button(label="Science Mode", width=60)
# Create button for "threshold" mode
threshold_button = Button(label="Threshold Mode", width=60)
# Assign handlers to buttons and toggles(glyphs in general) a.k.a. when button is clicked, this method will be called
threshold_button.on_click(threshold_button_handler)
button.on_click(science_button_handler)
science_button.on_click(science_button_handler)
exposure_textbox = TextInput(title="Enter the desired exposure time in ms", value=str(settings_json['exposure']))
gain_textbox = TextInput(title="Enter the desired gain value", value=str(settings_json['gain']))

exposure_textbox.on_change('value', exposure_handler)
gain_textbox.on_change('value', gain_handler)
# prepare some data
img = cv2.imread("Images/Threshold.tiff")
img = img[:, :, 0] # Make image mono
img = np.flipud(img) # Flip right-side up

# output to static HTML file (with CDN resources)
output_file("CMOS_Sensor.html", title="BokehGUI.py example", mode="cdn")
# Assign tools that can be used in the figure
TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"
# Create ColumnDataSource for image
source = ColumnDataSource(data=dict(image=[img]))

# create a new plot with the tools above, and explicit ranges
p = figure(tools=TOOLS, x_range=(0, img.shape[0]), y_range=(0, 255))
# add a circle renderer with vectorized colors and sizes
p.image(image='image', source=source, x=0, y=0, dw=img.shape[1], dh=img.shape[0], palette='Greys256')

# Assign handlers to buttons and toggles (glyphs in general) a.k.a. when glyph is modified, call this method
save_jpg_toggle.on_change("active", save_jpg_handler)
save_np_toggle.on_change("active", save_np_handler)
threshold_toggle.on_change("active", threshold_handler)
multi_toggle.on_change("active", multi_handler)
find_threshold_bool_slider.on_change("value", find_threshold_bool_handler)
run_button.on_change("active", run_button_handler)
num_images_slider.on_change("value", num_images_handler)
std_slider.on_change("value", std_handler)
save_settings_button.on_click(save_settings)
load_settings_button.on_click(load_settings)
packet_destination_textbox.on_change("value", textbox_handler)
# Display GUI
curdoc().add_root(
    row(column(save_np_toggle, save_jpg_toggle, threshold_toggle, multi_toggle, find_threshold_bool_slider, std_slider,
               num_images_slider, exposure_textbox, gain_textbox, run_button, p, slider, button), column(threshold_button, science_button, packet_destination_textbox, load_settings_button, save_settings_button)))
