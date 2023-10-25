# Import custom references from the "References" module.
from .References import References
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2lab
import numpy as np

class LoadData(References):

    def __init__(self):
        """
        Initialize the data loader.
        VGG16 expects an image of 3 dimensions with size 224x224 as input.
        In preprocessing, images are scaled to 224x224 instead of 256x256.
        """
        # Normalize images - divide pixel values by 255
        train_datagen = ImageDataGenerator(rescale=1. / 255)

        # Load and preprocess images from the specified directory.
        self.train = train_datagen.flow_from_directory(self.path, target_size=(224, 224), batch_size=100, class_mode=None)

    def getData(self):
        """
        Convert RGB images to LAB color space.
        LAB image is a grayscale image in the L channel, and all color information is stored in the A and B channels.
        :return: X (L channel), Y (normalized A and B channels)
        """
        X = []  # L channel
        Y = []  # Normalized A and B channels
        for img in self.train[0]:
            try:
                # Convert the RGB image to LAB color space.
                lab = rgb2lab(img)

                # Extract the L channel (luminance) and append it to X.
                X.append(lab[:, :, 0])

                # Extract the A and B channels (color information) and normalize the values.
                Y.append(lab[:, :, 1:] / 128)  # A and B values range from -127 to 128,
                # so we divide the values by 128 to restrict values to between -1 and 1.
            except:
                print('Error processing an image.')

        # Convert the lists to NumPy arrays for compatibility with the model.
        X = np.array(X)
        Y = np.array(Y)

        # Add an additional channel to the grayscale images to make them compatible with the model.
        X = X.reshape(X.shape + (1,))  # Make dimensions the same for X and Y.

        return X, Y
