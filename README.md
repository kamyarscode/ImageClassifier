<h1> Convolutional Neural Network - 5 environments -v1.0 </h1>

CNN that can predict an image to be a building, forest, glacier, mountain, sea, or street with ~90% accuracy

<h2> Structure </h2>

The images in this dataset are 150x150x3 and are fairly complex and can be computationally heavy to process. There are around 14,000 training images and 3,000
for testing.

The current implementation utilizes the idea that in the primary feature detection layers, there should be a gradual increase in the number of
neurons used per layer to detect more complex features in the later stages, and the classification layer should consist of fully-connected
layers that gradually decrease to reach 6 outputs.

At its current standing, it can accurately and precisely predict random and unseen inputs 90% of the time, so this model is fairly 
consistent. As there are currently no proven analytical approaches to determining best configurations, this model was created through
experimentation.

CUDA was used to increase the speed of training through parallel processing, but the model took nearly ~40 minutes to train 25 generations while consuming almost all available RAM. 
Any more layers in this current configuration would have lead to allocation errors that prematurely stopped training. 

<h2> Dataset </h2>

The Kaggle dataset Intel Image Classification was used.
https://www.kaggle.com/puneet6060/intel-image-classification

<h2> To Do</h2>

* Incorporate compiled numpy library more to increase performance

* Experiment with different number of neurons and classification layers to increase precision 

* Implement graphical analysis of loss and accuracy in different generations

* Test aerial angles of images
