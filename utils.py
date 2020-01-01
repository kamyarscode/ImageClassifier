import os
from keras.preprocessing import image
import matplotlib.pyplot as plt
import time

im_path = f'{os.getcwd()}\\images\\test\\'
compare = (150, 150)
num_of_folders = len(os.listdir(im_path))

def find_dimensions(path): # used to find images in training data with dimensions not 150, 150
    for folder in range(0,num_of_folders):
        image_list = [img for img in os.listdir(path + str(folder)) if img.endswith('.jpg')]
        for number in range (0, len(image_list)):
            im = image.load_img(im_path + str(folder) + '\\' + image_list[number])
            size = im.size
            if (size != compare):
                print (f'{image_list[number]} in {im_path}{folder} has file size of : {size}')


def input_img(dir, number):
    img = image.load_img(dir + f'{number}.jpg')
    img = img.resize((150,150))

    return image.img_to_array(img)


#find_test_dim()
start = time.time()
il = [img for img in os.listdir(im_path) if img.endswith('.jpg')]
print (il)
end = time.time()

print ('time: ', end - start)