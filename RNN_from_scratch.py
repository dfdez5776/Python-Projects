import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


#Load in training MNIST dataset (from kaggle)
data = pd.read_csv('train.csv')

#Convert data into numpy arrays

data = np.array(data)

#Notes on the data
#Each row is an example digit from 0 to 9 (10 classes). There are 784 pixels in each image 
#so there are 784 columns per row, corresponding to the pixel colorations of each image. Pixel 
#colors vary from 0 - 255. We transpose the matrix so each column is an example


#Model 1: 3 layers (input, hidden,  output), 10 nodes

#Separate into cross-validation
r, c = data.shape
#shuffle
np.random.shuffle(data)

data_cross = data[0:1000].T
Y_cross = data_cross[0]
X_cross = data_cross[1:c]

data_train = data [1000:r].T
Y_train = data_train[0]
X_train = data_train[1:c]

iterations = 500
alpha = 0.1

#Initialize parameters

def init_params():
    #Randomly initialize weights
    W1 = np.random.rand(10, 784) - 0.5
    #Randomly initialize bias
    b1 = np.random.rand(10,1) - 0.5
    W2 = np.random.rand(10,10) - 0.5
    b2 = np.random.rand(10,1) - 0.5
    return W1, b1, W2, b2

def ReLu(X):
    return np.maximum(0,X)

def softmax(X):
    return np.exp(X) / np.sum(np.exp(X))

def forward(W1, b1, W2, b2, X):
    #apply first layer weight to input, add bias
    X1= W1.dot(X) + b1
    #Activation function
    A1 = ReLu(X1)
 
    #apply output layer weight to output, add bias
    X2 = W2.dot(A1) + b2
    #Activation function 
    A2 = softmax(X1)
    return X1, A1, X2, A2

def categorize_examples(Y):
    #creates matrix of categories
    one_hot = np.zeros((Y.size, Y.max() + 1))
    one_hot[np.arange(Y.size), Y] = 1
    one_hot = one_hot.T
    return one_hot

def dRELU(X):
    if X > 0:
        return 1
    elif X <= 0:
        return 0

def back_prop(X1, A1, X2, A2, W2, X, Y):
    m = Y.size
    one_hot_y = categorize_examples(Y)
    #loss of output
    dX2 = A2 - one_hot_y
    #derivative of loss with respect to weights in layer 2
    dW2 = 1/ m * dX2.dot(A1.T)
    #average of absolute error
    db2 = 1/ m * np.sum(dX2,2)

    #loss in layer 1
    dX1 = W2.T.dot(dX2) * dRELU(X1)

    #derivative of 1st layer loss w/r to weights in layer 1
    dW1 = 1 / m * dX1.dot(X.T)
    #average of aboslute error in laber 2
    db1 = 1 / m * np.sum(dX1, 2)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, X, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1 
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def predict(A2):
    return np.argmax(A2, 0)

def eval_performance(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def train(X, Y, iter, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iter):
        X1, A1, X2, A2 = forward(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(X1, A1, X2, A2, W2, X, Y)
        W1, b1, W2, b2  =  update_params(1, b1, W2, b2, X, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print("Iteration:", i)
            print("Accuracy:", eval_performance(predict(A2), Y))
    return W1, b1, W2, b2 


W1, b1, W2, b2 = train(X_train, Y_train, iterations, alpha )

def make_predictions( X, W1, b1, W2, b2 ):
    _, _, _, A2 = forward(W1, b1, W2, b2, X)
    predictions = predict(A2)
    return predictions


#To test any individual image
def test(idx, W1, b1, W2, b2 ):
    current_image = X_train[:, idx, None]
    prediction = make_predictions(X_train[:, idx, None], W1, b1, W2, b2 )
    label = Y_train[idx]
    print("Prediction:", prediction)
    print("Label:", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation = 'nearest')
    plt.show()






#Model 2: 3 layers, 10 nodes
