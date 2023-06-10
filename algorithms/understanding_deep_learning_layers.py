############################################################################
#                         Understanding Layers
############################################################################

"""
In deep learning, layers are the fundamental units of neural networks. A layer
is a collection of "neurons" or "nodes" operating together at the same depth
level within a neural network. The inputs are processed through these layers to
generate outputs. There are three types of layers in a typical
neural network: input layer, hidden layers, and output layer.

Input Layer: The input layer is the very first layer in any neural network.
It is responsible for receiving all the inputs and forwarding them to the next
layer (the first hidden layer, if present). The input layer does not perform
any computations, its primary job is to pass the information to the next layer.
The number of nodes in this layer equals the number of features in the input
dataset_sites.

Hidden Layers: These are the layers between the input and output layers.
The term "hidden" simply means that they are not visible as input or output
layers. Each hidden layer extracts some kind of information from the input dataset_sites,
and the complexity of the information generally increases with each subsequent
layer. Hidden layers perform transformations on their inputs using weights,
biases, and activation functions. Each neuron in a hidden layer will receive
inputs from all the neurons of the previous layer, applies the weights, adds
the bias, applies the activation function, and sends this output to all the
neurons in the next layer.

Output Layer: The final layer is the output layer. This layer transforms the
dataset_sites from the hidden layers into the final output format. Common transformations
include classification (where the output is a probability distribution over
classes) and regression (where the output is a real number). The number of
nodes in the output layer corresponds to the number of output values.

Apart from these, there are several specialized types of layers used in different
types of neural networks:

Dense (Fully Connected) Layers: Every neuron in a dense layer receives input
from every neuron in the previous layer.

Convolutional Layers: These are the major building blocks of convolutional neural
networks (CNNs), which are primarily used for image processing. These layers
apply a convolution operation to the input.

Pooling Layers: These layers are also used in CNNs where they reduce the
spatial dimensions (width and height) of the input volume. They are used for
reducing the computational complexity and to control overfitting.

Recurrent Layers: These layers are used in recurrent neural networks (RNNs),
which are designed to recognize patterns in sequences of dataset_sites, such as text,
and handwriting.

Normalization Layers: These layers are used to standardize the inputs. Example:
Batch Normalization.

Dropout Layers: These layers apply regularization. In a dropout layer, a
random set of activations is set to zero within each training mini-batch,
which helps prevent overfitting.

Remember that the design of a neural network (which includes the number of
layers, the type of layers, the number of neurons in the layers, etc.) is
typically decided based on the problem at hand, and can be quite complex.
Understanding how each type of layer works and what kind of dataset_sites or problem
they are best suited for is crucial in designing an effective neural network.
"""

############################################################################
#                             Examples
############################################################################
"""
import Sequential from keras.models if you are to use these 
layers to construct a model like this:

This creates a connection between all the neurons in the layers.
"""
from keras.models import Sequential
model = Sequential()


"""
Dense (Fully Connected) Layers

In this example, the first layer has 64 nodes and uses the ReLU (Rectified 
Linear Unit) activation function. The second layer has 1 node (which could 
serve as an output for binary classification) and uses the sigmoid activation 
function.
"""
from keras.layers import Dense
model.add(Dense(64, activation='relu', input_dim=50))
model.add(Dense(1, activation='sigmoid'))

"""
Convolutional Layers

This creates a convolutional layer with 32 output filters and a kernel size 
of 3x3, using the ReLU activation function. It expects input tensors to be 
64x64 with 3 channels.
"""
from keras.layers import Conv2D
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))

"""
Pooling Layers

This layer will perform max pooling operation using 2x2 windows.
"""
from keras.layers import MaxPooling2D
model.add(MaxPooling2D(pool_size=(2, 2)))

"""
Recurrent Layers

This adds an LSTM (Long Short Term Memory) layer with 32 memory units.
"""
from keras.layers import LSTM
model.add(LSTM(32))

"""
Normalization Layers

This will normalize the activations of the previous layer at each batch 
(i.e. it applies a transformation that maintains the mean activation close to 
0 and the activation standard deviation close to 1).
"""
from keras.layers import BatchNormalization
model.add(BatchNormalization())

"""
Dropout Layers

This applies dropout to the input, randomly setting half (50%) of the input 
units to 0 at each update during training time to prevent overfitting.

Remember to import Sequential from keras.models if you are to use these layers 
to construct a model like this:
"""
from keras.layers import Dropout
model.add(Dropout(0.5))


