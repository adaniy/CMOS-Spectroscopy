import numpy as np
import argparse
import datetime
import PySpin
import os
import cv2
import time

def setup():
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
    cam = cam_list.GetByIndex(0)
    cam.Init()
    print("Camera firmware version: " + cam.DeviceFirmwareVersion.ToString())
    # load default config
    cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
    cam.UserSetLoad()
    # set acquisition to continuous, turn off auto exposure, set the frame rate
    # Camera Settings
    # Set Packet Size
    cam.GevSCPSPacketSize.SetValue(9000)
    cam.DeviceLinkThroughputLimit.SetValue(100000000)
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
    return cam, system

def find_threshold(cam):
    t0 = time.time()
    save_folder = "C:\\Users\\habraha1\\Documents\\Images\\"
    print(cv2.__version__)
    image = cam.GetNextImage()
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


def convert_images(images, temp_threshold):
    for threshold_image in images:
        threshold_image[threshold_image > temp_threshold] = 255
        threshold_image[threshold_image <= temp_threshold] = 0


def capture_images(cam, num_images):
    images = []
    for i in range(0, num_images):
        image = cam.GetNextImage()
        images.append(image)
    return images


def save_images_jpg(images):
    i = 0
    for temp_img in images:
        cv2.imwrite("Image " + str(i) + ".jpg", temp_img)
        i += 1


def save_images_np(images):
    n = 0
    for np_img in images:
        np.save("NpArray " + str(datetime.time()), np_img)
        n += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parses arguments")
    parser.add_argument('path', metavar='P', type=str, nargs='1', help='path to save file')
    parser.add_argument('--jpg', '--jpg', help='save image as jpg')
    parser.add_argument('--np', '--np', help='save image as np array')
    parser.add_argument('--t', '--threshold', help="threshold image")
    parser.add_argument('num', metavar='N', type=int, nargs=1, help='number of images to collect')
    args = parser.parse_args()
    cam, system = setup()
    main_images = capture_images(cam, args.num)
    if args.jpg is not None:
        save_images_jpg(main_images)
    if args.np is not None:
        save_images_np(main_images)
    if args.threshold:
        threshold = find_threshold(cam)
        convert_images(main_images, threshold)
    if args.jpg is not None:
        save_images_jpg(main_images)
    if args.np is not None:
        save_images_np(main_images)
