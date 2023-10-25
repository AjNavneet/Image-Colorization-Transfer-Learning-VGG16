from keras.layers import Conv2D, UpSampling2D
from keras.models import Sequential
import tensorflow as tf

# Import custom references from the "References" module.
from .References import References

class DecoderLayers(References):
    def __init__(self):
        # Initialize the model as a sequential model.
        self.model = Sequential()
        # Configure the model architecture.
        self.__setModelArch()

    def __setModelArch(self):
        """
        Setting up the model Architecture
        :return: None
        """
        # Add a convolutional layer with 256 filters, a 3x3 kernel, ReLU activation, and same padding.
        # Input shape is (7, 7, 512).
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=(7, 7, 512)))

        # Add another convolutional layer with 128 filters and ReLU activation.
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

        # Upsample the feature maps by a factor of 2.
        self.model.add(UpSampling2D((2, 2)))

        # Add a convolutional layer with 64 filters and ReLU activation.
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

        # Upsample the feature maps by a factor of 2.
        self.model.add(UpSampling2D((2, 2)))

        # Add a convolutional layer with 32 filters and ReLU activation.
        self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

        # Upsample the feature maps by a factor of 2.
        self.model.add(UpSampling2D((2, 2)))

        # Add a convolutional layer with 16 filters and ReLU activation.
        self.model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))

        # Upsample the feature maps by a factor of 2.
        self.model.add(UpSampling2D((2, 2)))

        # Add the final convolutional layer with 2 filters, tanh activation, and same padding.
        self.model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

        # Compile the model with the Adam optimizer, mean squared error (MSE) loss, and accuracy metric.
        self.model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])

    def fit(self, vggfeatures, Y):
        """
        Fitting up the model for training
        :param vggfeatures: Features extracted from VGG model
        :param Y: Target data for training
        :return: None
        """
        # Fit the model with the provided VGG features and target data.
        self.model.fit(vggfeatures, Y, verbose=1, epochs=2000, batch_size=16)

        # Save the trained model.
        self.model.save(self.ROOT_DIR + self.SAVE_MODEL)

    def load_model(self):
        """
        Loading Trained Model
        :return: The loaded model
        """
        # Load the saved model from the specified location and return it.
        self.model = tf.keras.models.load_model(self.ROOT_DIR + self.SAVE_MODEL,
                                               custom_objects=None,
                                               compile=True)
        return self.model
