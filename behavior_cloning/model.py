import pickle
import numpy as np
import math
import tensorflow as tf
tf.python.control_flow_ops = tf
import csv
import numpy
import re
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def process_image(img):
    img = cv2.resize(img, (32, 16)) 
    return img

def get_data(csv_file, path, flip = False):
    with open(csv_file, 'r') as f:
        car_images = []
        steering_angles = []
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            steering_center = float(row[3])
            center_img_name = re.sub("/Users/qianliu/Downloads/IMG/","",row[0]).lstrip()
            img_center = process_image(np.asarray(Image.open(path + center_img_name)))
            img_copy = img_center.copy()
            flip_img_center = cv2.flip(img_copy,1)
            flip_steering_center = -steering_center
            # add images and angles to data set
            car_images.extend([img_center])
            steering_angles.extend([steering_center])
            if flip == True:
                car_images.extend([flip_img_center])
                steering_angles.extend([flip_steering_center])
        
        print(len(car_images))	
        return car_images, steering_angles

def preprocessing():
	csv_file_3, path_3 = '/home/carnd/behavior-cloning/CURVES/driving_log.csv', "/home/carnd/behavior-cloning/CURVES/IMG/"
	car_images_3, steering_angles_3 = get_data(csv_file_3, path_3)
	csv_file_4, path_4 = '/home/carnd/behavior-cloning/RECOVERY/driving_log.csv', "/home/carnd/behavior-cloning/RECOVERY/IMG/"
	car_images_4, steering_angles_4 = get_data(csv_file_4, path_4)
	csv_file_5, path_5 = '/home/carnd/behavior-cloning/data/driving_log.csv', "/home/carnd/behavior-cloning/data/"
	car_images_5, steering_angles_5 = get_data(csv_file_5, path_5, True)
	csv_file_6, path_6 = '/home/carnd/behavior-cloning/recovery_1/driving_log.csv', "/home/carnd/behavior-cloning/recovery_1/IMG/"
	car_images_6, steering_angles_6 = get_data(csv_file_6, path_6)
	csv_file_7, path_7 = '/home/carnd/behavior-cloning/recovery_2/driving_log.csv', "/home/carnd/behavior-cloning/recovery_2/IMG/"
	car_images_7, steering_angles_7 = get_data(csv_file_7, path_7)
	csv_file_8, path_8 = '/home/carnd/behavior-cloning/recovery_3/driving_log.csv', "/home/carnd/behavior-cloning/recovery_3/IMG/"
	car_images_8, steering_angles_8 = get_data(csv_file_8, path_8)
	csv_file_9, path_9 = '/home/carnd/behavior-cloning/recovery_4/driving_log.csv', "/home/carnd/behavior-cloning/recovery_4/IMG/"
	car_images_9, steering_angles_9 = get_data(csv_file_9, path_9)
	car_images = car_images_3 + car_images_4 + car_images_5 + car_images_6 + car_images_7 + car_images_8 + car_images_9
	steering_angles = steering_angles_3 + steering_angles_4 + steering_angles_5 + steering_angles_6 + steering_angles_7 + steering_angles_8 + steering_angles_9
	car_images = np.array(car_images)
	steering_angles = np.array(steering_angles)
	return car_images, steering_angles

def model(X_train, y_train, X_test, y_test):
	pool_size = (1,1)
	model = Sequential()
	## cropping images
	model.add(Cropping2D(cropping=((5,2), (0,0)), input_shape=(16,32,3), dim_ordering='tf'))
	## normalization layer
	model.add(Lambda(lambda x: (x / 255.0) - 0.5))
	# 1st layer
	model.add(Convolution2D(24, 5, 5, input_shape=(9, 32, 3)))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(0.5))
	model.add(Activation('relu'))
	
	# 2nd layer
	model.add(Convolution2D(36, 5, 5, border_mode = 'same'))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(0.5))
	model.add(Activation('relu'))

	# 3rd layer
	model.add(Convolution2D(48, 3, 3, border_mode = 'same'))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(0.5))
	model.add(Activation('relu'))
	
	model.add(Flatten())

	## 4 FC layers
	model.add(Dense(100))
	model.add(Activation('relu'))
	model.add(Dense(80))
	model.add(Activation('relu'))
	model.add(Dense(60))
	model.add(Activation('relu'))
	model.add(Dense(40))
	model.add(Activation("relu"))
	model.add(Dense(1))

	## numerical not categorical, optimizer, 
	model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
	model.fit(X_train, y_train, batch_size= 128, nb_epoch= 5, validation_split = 0.2, shuffle = True)
	model.save('my_model.h5') 
	score = model.evaluate(X_test, y_test, verbose = 0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
def main():
	car_images, steering_angles = preprocessing()
	car_images, steering_angles = shuffle(car_images, steering_angles)
	X_train, X_test, y_train, y_test = train_test_split(car_images, steering_angles, test_size=0.2)
	model(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
	main()
