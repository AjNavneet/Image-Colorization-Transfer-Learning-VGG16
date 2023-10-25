# Import necessary libraries for image colorization using Autoencoders and transfer learning with VGG.

from keras.preprocessing.image import img_to_array, load_img
import os

# Import custom modules for the machine learning pipeline.
from ML_Pipeline.EncoderLayers import EncoderLayers
from ML_Pipeline.LoadData import LoadData
from ML_Pipeline.DecoderLayers import DecoderLayers
from ML_Pipeline.Inference import Inference

"""
Training Phase: This section focuses on model training.
"""

def trainingPhase():

    # Load and prepare the data for training.
    datagen = LoadData()
    X, Y = datagen.getData()
    
    # Set up the Encoding Layer and extract features from the data.
    encoder_layer = EncoderLayers()
    encoded = encoder_layer.getfeatures(X)

    # Set up the Decoding Layer and train it with the encoded features and target data.
    decoder_layer = DecoderLayers()
    decoder_layer.fit(encoded, Y)

"""
Inference Phase: This section focuses on using the trained model for image colorization.
"""

def inferencePhase():
    # Set up the Encoding Layer for inference.
    encoder_layer = EncoderLayers()
    encode_model = encoder_layer.newmodel

    # Set up the Decoding Layer and load the trained model.
    decoder_layer = DecoderLayers()
    model = decoder_layer.load_model()

    # Define the path to the test images.
    testpath = '../Input/dataset/test/test/'
    files = os.listdir(testpath)
    print("Test files:", files)
    inf = Inference()

    # Perform inference on test images and produce colorized outputs.
    for idx, file in enumerate(files):
        print("Processing file:", file)
        # Load and process the test image.
        test = img_to_array(load_img(testpath + file))
        inf.processImg(idx, test, encode_model, model)

### Training the model ###
trainingPhase()

### Perform inference ###
inferencePhase()
