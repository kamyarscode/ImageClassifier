from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout
from keras.preprocessing import image
import numpy as np
import os

#.shape gives us (total images, pixel len, pixel wid, dimensions(3 = RGB))
#print (x_train.shape)
# print (x_train[0])
# show = plt.imshow(x_train[0])
# plt.show()

class Model():

    def __init__(self):
        self.model = Sequential()

    def preprocess(self, dir):
        processed_list = []
        num_of_folders = len(os.listdir(dir))
        labels_array = np.empty(shape=(0, 1))  # create empty array with 1 column
        for num in range(0, num_of_folders):
            image_list = [img for img in os.listdir(dir + str(num)) if img.endswith('.jpg')]
            for img_name in range(0, len(image_list)):
                loaded_image = image.load_img(dir + str(num) + '\\' + str(image_list[img_name]), grayscale=False)
                loaded_image = loaded_image.resize((150, 150)) #resize every image to 150x150
                process_img = image.img_to_array(loaded_image) #instead of image.img_to_array()
                processed_list.append(process_img / 255) #normalize rgb values by giving it value from 0 to 1
            labels = np.full((len(image_list), 1), num)  # create matrix of size 16x1 filled with label of 0 through 5
            labels_array = np.concatenate((labels_array, labels))  # add previous label arrays together

        return np.array(processed_list), labels_array

    def build(self):
        #Initially have 150x150x3 = 67,500 input neurons without convolution and restructuring
        #kernel_size = 3x3 filter matrix to convolve with the input to create feature maps.
        #Start with 32 feature maps and gradually increase to detect more complex features
        self.model.add(Conv2D(32, (3,3), activation = 'relu', padding = 'same', input_shape = (150,150,3)))
        self.model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))
        self.model.add(MaxPooling2D(pool_size = (2,2))) #pooling uses MAX operation get highest numbers in 2x2 matrix. - dimension transformation
        self.model.add(Dropout(.25))

        self.model.add(Conv2D(64,(3,3), activation = 'relu', padding = 'same'))
        self.model.add(Conv2D(100,(3,3), activation = 'relu'))
        self.model.add(MaxPooling2D(pool_size = (2,2)))
        self.model.add(Dropout(.25))

        # self.model.add(Conv2D(64,(3,3), activation = 'relu', padding = 'same'))
        # self.model.add(Conv2D(32,(3,3), activation = 'relu'))
        # self.model.add(MaxPooling2D(pool_size = (2,2)))
        # self.model.add(Dropout(.25))

        #classification layers
        self.model.add(Flatten()) #collapse dimensions
        #self.model.add(Dense(200, activation='relu'))
        self.model.add(Dense(120, activation = 'relu')) #first fully connected layer
        self.model.add(Dense(50, activation='relu'))
        #self.model.add(Dense(20, activation='relu')) dropped generation 1 accuracy from 40% to ~20%
        self.model.add(Dropout(.5))
        self.model.add(Dense(6, activation = 'softmax')) #final fully-connected layer leading to 6 outputs.

        return self.model

