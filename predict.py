from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import random as r
import os


LABELS = ['Building','Forest','Glacier','Mountain','Sea','Street']
VALIDATIONDIR = f'{os.getcwd()}\\images\\validation\\'
DIMENSIONS = (150,150) #tuple of what dimension all images should be

model = load_model('CNNModel.h5')

def main():
    count, total = 0, 0
    for folder in range (0,6):
        image_list = [im for im in os.listdir(VALIDATIONDIR + f'{folder}') if im.endswith('.jpg')]
        for num in range (0,20):
            randnum = r.randint(0, len(image_list) - 1)
            img = image.load_img(VALIDATIONDIR + f'{folder}\\' + image_list[randnum]) #load random image as an Image instance
            if (img.size != DIMENSIONS):
                img = img.resize(DIMENSIONS) #convert to 150x150 image
            ar = np.array(img) #convert Image to 3D numpy array

            predictions = model.predict(np.array([ar,])) #convert img to 4D numpy array before taking as input to model
            if predictions[0][folder] == 1:
                count += 1
            total += 1

            #index = np.argsort(predictions)
            print (f'Predicting : {folder}\\{image_list[randnum]}')
            print(predictions, ' - Labeled: ', LABELS[folder])
            print (count)

    print (count, ' out of ', total)




if __name__ == '__main__':
    main()