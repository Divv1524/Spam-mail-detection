# Spam-mail-detection

Sure! Let's go through the code step-by-step to understand what each part does:

1)Import the necessary libraries first

numpy is used for numerical operations.
matplotlib.pyplot is used for data visualization.
tensorflow.keras.datasets.mnist provides access to the MNIST dataset.
tensorflow.keras.models.Sequential is used to create a sequential model (a linear stack of layers).
tensorflow.keras.layers.Dense is used to create a fully connected layer in the neural network.
tensorflow.keras.layers.Flatten is used to flatten the input image data before passing it to the fully connected layers.
tensorflow.keras.utils.to_categorical is used to one-hot encode the target labels.

2)Load the MNIST dataset:
The mnist.load_data() function loads the MNIST dataset. It returns two tuples: (X_train, y_train) containing the training images and their corresponding labels, and (X_test, y_test) containing the test images and their corresponding labels.

3)Preprocess the data:
The images are reshaped to have a single channel (grayscale) and normalized by dividing by 255 to scale the pixel values between 0 and 1. The labels are one-hot encoded using to_categorical to convert the class labels into binary vectors.

4)Build the neural network model:
We create a sequential model and add layers to it. The first layer is a Flatten layer, which converts the 28x28 input images into a 1D array. Then, we add two fully connected (Dense) layers with 128 and 64 units, respectively, with ReLU activation functions. The last layer has 10 units with a softmax activation function, which is used for multi-class classification.

5)Compile the model
6)Train the model
7)Evaluate the model
8)Make predictions
9)Visualize the predictions

We visualize the first 10 test images along with their predicted labels and true labels to see how well the model performs.

This code provides a basic example of recognizing handwritten digits using a simple neural network model.
