# Image Colourization using transfer learning - VGG16 model

## Business Context
Image colorization is the process of transforming grayscale images into colorized images. This project focuses on using autoencoders and the VGG16 model to achieve this. 

Image colorization finds applications in various fields, including Medical Microscope, Medical Imagery, Denoising old images, Night Vision Cameras, and more.

---

## Data Description
We will be using a landscape image dataset consisting of:
- Training: Approximately 7,000 RGB images
- Testing: 5 grayscale images

---

## Aim
To build a Keras model that converts grayscale images to colored images.

---

## Tech Stack
- Language: `Python`
- Libraries: `NumPy`, `Pandas`, `TensorFlow`, `Keras`

---

## Approach
1. Mount the drive for loading input data images.
2. Import the necessary libraries.
3. Import and initialize the VGG16 model.
4. Initialize the ImageDataGenerator to rescale the images.
5. Convert the RGB to LAB format.
6. Create a sequential model and check its summary.
7. Compile the model with the appropriate optimizer, loss, and performance metric.
8. Fit the model and train it for 2,000 epochs with a defined batch size.
9. Save the model.
10. Use the saved model to make predictions on the test images.
11. Check the predicted test images.

---

## Modular Code Overview

1. **Input**: Contains the data for analysis (training and test images).
2. **Src**: Contains modularized code for each step, including `Engine.py` and `ML_Pipeline` folder.
3. **Output**: Contains the best-fitted model and prediction results.
4. **Lib**: Reference folder with IPython notebooks and reference content.

---

## Key Concepts Explored

1. Understanding the business context.
2. Implementing autoencoders.
3. Leveraging transfer learning and backbone concepts.
4. Exploring the VGG16 model architecture.
5. Using the VGG16 model.
6. Rescaling images using ImageDataGenerator.
7. Converting RGB to LAB format.
8. Building a sequential model.
9. Compiling and training model with suitable parameters.
10. Saving the trained model for future use.
11. Interpreting the generated results.
12. Making predictions on test data using the trained model.

---

## Getting Started

There are two ways to execute the end to end flow.

- Modular Code
- IPython (Google Colab)

### Modular code

- Create virtualenv
- Install requirements `pip install -r requirements.txt`
- Modify `Engine.py` based on the mode that you are training on "Training" / "Inference"
- Run Code `python Engine.py`

---




