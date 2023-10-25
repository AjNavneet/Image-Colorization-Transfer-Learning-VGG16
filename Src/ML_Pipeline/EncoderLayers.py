from keras.models import Sequential
from skimage.color import gray2rgb
import numpy as np

# Import the VGG16 model from Keras.
from keras.applications.vgg16 import VGG16
# Import custom references from the "References" module.
from .References import References

class EncoderLayers(References):
    def __init__(self):
        """ Initializing the Encoder"""
        # Initialize the VGG16 model as the feature extractor.
        self.vggmodel = VGG16()
        # Initialize a new sequential model for feature extraction.
        self.newmodel = Sequential()
        # Create the layers for feature extraction.
        self.layers()

    def layers(self):
        "Replacing the encoder part with Feature Extractor of VGG"
        for i, layer in enumerate(self.vggmodel.layers):
            if i < 19:  # Include only the first 19 layers for feature extraction
                self.newmodel.add(layer)
        # Display a summary of the new model's architecture.
        self.newmodel.summary()
        # Set all layers in the new model to be non-trainable to retain pre-trained weights.
        for layer in self.newmodel.layers:
            layer.trainable = False
        return self.newmodel

    def getfeatures(self, X):
        # Prepare input data for feature extraction using VGG16.
        vggfeatures = []
        for i, sample in enumerate(X):
            # Convert a grayscale image to RGB format.
            sample = gray2rgb(sample)
            # Reshape the sample to match VGG16's input shape.
            sample = sample.reshape((1, 224, 224, 3))
            # Extract features from the sample using the new model.
            prediction = self.newmodel.predict(sample)
            # Reshape the prediction to the expected shape (7x7x512).
            prediction = prediction.reshape((7, 7, 512))
            vggfeatures.append(prediction)
        # Convert the list of feature maps to a NumPy array.
        vggfeatures = np.array(vggfeatures)
        return vggfeatures
