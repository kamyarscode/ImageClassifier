from keras.utils import to_categorical
import os
import time

from model import Model

TRAIN_SAM = f'{os.getcwd()}\\images\\train_sample\\'
TEST_SAM = f'{os.getcwd()}\\images\\test_sample\\'
TRAINDIR = f'{os.getcwd()}\\images\\train\\'
VALIDATIONDIR = f'{os.getcwd()}\\images\\validation\\'
#TESTDIR = f'{os.getcwd()}\\images\\test\\'
EPOCH = 25
LRATE = .1
BATCH_SIZE = 100

start = time.time()

m = Model()
model = m.build()

# x_ is img info, y_ is labels
train, train_label = m.preprocess(TRAINDIR)
validate, validate_label = m.preprocess(VALIDATIONDIR)

train_cat = to_categorical(train_label, 6)  # 6 labels = 6 output neurons, change this later to variable instead of hard code
validate_cat = to_categorical(validate_label, 6)

print ('shape of training data: ', train.shape) #confirm dimensions of data
print ('shape of test data: ', validate.shape)

print ('------------ MODEL COMPILING ------------')
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print ('------------ COMPILING COMPLETE ------------ \n ------------ MODEL SUMMARY: ------------')
model.summary()

print ('------------ MODEL FITTING: ------------')
model.fit(train, train_cat,
             batch_size = BATCH_SIZE,
             epochs = EPOCH,
             verbose = 1,
             validation_data = (validate, validate_cat)
             )
print ('------------ FITTING COMPLETE ------------')

print ('------------ EVALUATION OF MODEL: ------------')
model.evaluate(validate, validate_cat)
print ('------------ EVALUATION COMPLETE ------------')

print ('------------ SAVING MODEL ------------')
json = model.to_json()
with open ('CNNmodeljson2.json', 'w') as json_file:
    json_file.write(json)
model.save('CNNModel2.h5')
model.save_weights('imageWeights2.h5')
print ('------------ MODEL SAVED ------------')

end = time.time()

print ('time to completion: ', (end - start))


