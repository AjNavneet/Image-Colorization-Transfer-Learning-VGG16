from skimage.color import rgb2lab, lab2rgb, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np

# Import custom references from the "References" module.
from .References import References

class Inference(References):

    def processImg(self, idx, test, newmodel, model):
        """
        Process an image and predict the colorized output.
        :param idx: Index or identifier for the processed image.
        :param test: The input image to be colorized.
        :param newmodel: The feature extraction model (encoder).
        :param model: The colorization model (decoder).
        :return: None
        """
        # Resize the input image to the expected size (224x224).
        test = resize(test, (224, 224), anti_aliasing=True)

        # Normalize the resized image to the range [0, 1].
        test *= 1.0 / 255

        # Convert the RGB image to LAB color space.
        lab = rgb2lab(test)

        # Extract the L channel (luminance) from LAB color space.
        l = lab[:, :, 0]

        # Convert the L channel to a grayscale image with three channels (RGB).
        L = gray2rgb(l)
        L = L.reshape((1, 224, 224, 3))

        # Extract VGG features from the L channel using the feature extraction model (encoder).
        vggpred = newmodel.predict(L)

        # Predict the 'ab' channels (color information) using the colorization model (decoder).
        ab = model.predict(vggpred)

        # Scale the 'ab' channels by a factor of 128.
        ab = ab * 128

        # Create an array to hold the LAB image.
        cur = np.zeros((224, 224, 3))

        # Assign the L channel to the first channel of the LAB image.
        cur[:, :, 0] = l

        # Assign the predicted 'ab' channels to the second and third channels of the LAB image.
        cur[:, :, 1:] = ab

        # Convert the LAB image back to RGB color space and save the colorized image.
        imsave(self.ROOT_DIR + self.TEST_IMG + str(idx) + ".jpg", lab2rgb(cur))
