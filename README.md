# seals-cnn

Requires Tensorflow.

This project takes image datasets in tfrecord format and predicts meaningful semnatic landmarks, which are then used to build
point distribution models for image retrieval. 

To format your own dataset into a tfrecord file which is accepted by the CNN please use lfw_parse.py. lfw_parse.py does not take any command line arguments and you will need to change variables inside python file to match the file destination of your dataset and the preferred output name of the tfrecord file.

To train the CNN, run lfw_train.py and to evaluate data run lfw_eval.py. If you wish to change the training and evaluation datasets, you will need to change the global variables within lfw_input.py.

Neither of these modules take command line arguments at the moment, however each file does contain global variables that you can use to tweek model parameters.

When ran, lfw_eval.py will store landmarked versions of images in the evaluation set in the output-images folder and number them. lfw_eval.py will also print the numbers of the best 5 matching images to a few sample images in the evaluation dataset. Matching is done by forming two point ditribution models and combining them together into one point distribution model. The first point distribution model is built from the landmark coordinates themselves, while the second point ditribution model is built using the pixel patches around each landmark as feature vectors.

I didn't find the combined point distribution to provide very accurate results, although I didn't have time to play with many of the parameters.
 
