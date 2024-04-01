# Project Details:
This project is intended to be a modular playground for building neural nets.
I have coded this in plain C to get a clear, strong foundational understanding 
of Neural nets. All math used here was derived on paper (see documentation 
folder for images and calculations). 

# Demo:
To view a demo: 
1. please launch terminal and cd to "Neural Net C" folder containing all files.
2. Type "make" (this will require Makefile)
3. Type "./test"
This should display 10 images with prediction beneath. At the top you may find the 
accuracy of the neural net being used, which has been tested over 10k images. 


# Warning:
If you intend to change the activation functions, you may do so manually in the
neuralNet file. Alternatively, code your own helper functions and override the
function pointers in the NeuralNet struct. 


# Code Structure:
1. "backProp.c": This file allocates memory for training efficiency, and has the
                 implementation for the backpropagation algorithm.
2. "loadData.c": This file reads in MNIST data, stores neural nets, and can store them.
3. "neuralNet.c": allocated and initializes weights and biases.
4. "train.c": trains the neural net based on manual settings.
5. "test.c": test a model over 10k images, and provide a short demo.
