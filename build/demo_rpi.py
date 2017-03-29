import os
import time
import numpy as np
import scipy.misc
import pygame, sys
from pygame.locals import *
import pygame.camera
from keras.models import load_model
import RPi.GPIO as GPIO

green_pin = 23
red_pin = 18

GPIO.setmode(GPIO.BCM)
GPIO.setup(green_pin, GPIO.OUT, initial=False)
GPIO.setup(red_pin, GPIO.OUT, initial=False)


model = load_model("data/models/wider_network_more_data/best_weights.hdf5")
data_mean = np.load("data_mean.npy")


width = 640
height = 480

pygame.init()
pygame.camera.init()
cam = pygame.camera.Camera("/dev/video0", (width, height))


try:
	cam.start()
except SystemError as e:
	print("Please connect a USB camera to the RPi")
	print("Error was: {}".format(e))


try:
	coin = False
	seeing_since = 0
	while True:
	    # video feed
	    image = cam.get_image()
	    
	    # Image prediction
	    image = pygame.surfarray.array3d(image)
	    image = image.transpose((2, 1, 0))
	    center_x = image.shape[1] // 2
	    center_y = image.shape[2] // 2
	    image = image[:, center_x - 100:center_x + 100, center_y - 100: center_y + 100]
	    # scipy.misc.imsave('feed.jpg', image)

	    image = image.reshape((1, 3, 200, 200))
	    image = np.array(image, dtype="float32")
	    image /= 255.0
	    image -= data_mean
	    
	    prediction = model.predict(image)[0][1]
	    if prediction > 0.5 and not coin:
		seeing_since = 0
	    elif prediction > 0.5:
		seeing_since += 1
	    
	    if seeing_since == 5:
		os.system('omxplayer -o local /notebooks/deep/corrected_tribute_voice.mp3 &')	
		
	    coin = bool(prediction > 0.5)
	    print("Coin: {}".format(coin))
	    # now set the pin outputs based on the recognition
	    GPIO.output(18, not coin)
	    GPIO.output(23, coin)
	    time.sleep(0.1)
finally:	
	cam.stop()
