# seals-cnn

Requires Tensorflow.

This project takes image datasets in tfrecord format and predicts meaningful semnatic landmarks, which are then used to build
point distribution models for image retrieval. 

To format your own dataset into a tfrecord file which is accepted by the CNN please use lfw_parse.py. To train the CNN, run lfw_train.py and to evaluate data run lfw_eval.py. Neither of these modules take command line arguments at the moment, however each file does contain global variables that you can use to tweek parameters. If you wish to change the training and evaluation datasets, you will need to change the global variables within lfw_input.py.

Landmarked images are saved to the output-images folder. 
