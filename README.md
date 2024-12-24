# Neural_Network_Test
Creating a neural network from scratch, mostly to help me review some of the math of it.
Main implementation resides in NN_test.ipynb

## Summary of The Code:
- This is an implementation of a neural network form scratch with 2 layers that Is meant to classify numbers from 0 - 9 (10 possible classes)
- Uses matplotlib, numpy, and pandas
- A chunk of the tutorial was followed from: https://www.youtube.com/watch?v=w8yWXqWQYmU (mostly with  dealing with the backpropagation step) However a lot of the existing code needed to be refactored and changed to prevent specific errors from occuring
- Fixed the normalization of data issue, possible exploding gradient problem.

## How Neural Network works In this Case (Quick Summary Written by Me):
A neural is a specific type of machine learning method which inputs features and outputs a predictor (just like ERM or MLE) We try to minimize the loss to find the best
possible predictor which can be used on the testing set (estimate of the expected loss) that helps determine how well your outputted predictor performs on new data

## Forward Propagation
Forward propagation is moving from different layers of neurons to the next layer of neurons, this is done through taking the dot product of the feature * the weight vector which 
will have a connection to the next layers neuron ex) (x1 ... xn) * (w0,1 ... w0,n) + b will all have a connection to (a(1)) and then another (x1 ... xn) * (w1,1 ... w1,n) + b(1) to a(2)

After the dot product we get a value of Z which we then use an activation function h() to get a(1) .. a(d) which will be all the neurons in that specific layer.
For the layer 1 we use the ReLU fucntion which takes the maximum of 0 and Z. In the last layer (Layer 2) we use the softmax function. The softmax function is important in classification
as it allows us to represent the probabilities between 0 and 1. In this case it will be the max of how likely that a specific number is to be the actual number that is represented in the image.

## BackPropagation
This is the process of running gradient descent on the neurons to try and find values of W that minimize the loss of all the w vectors that point towards the nodes in the current layer.
Through the chain rule we can get the first derivative with respect to w. Once we do this we have to run gradient descent which is back propagation from the current layer towards previous layers.
REMEMBER that W is a vector

```
| W0,0 ... W0,n |    |a1(L - 1)|
| W1,0 ... W1,n |    |a2(L - 1)|
| W2,0 ... W2,n | *  |a3(L - 1)|
| ...           |    |...      |
| Wd,0 ... Wd,n |    |ad(L - 1)|
```

The update rule for each is 
w = w - step_size * dw1 (with respect to the current index of the w vector that we want to minimize) 
This rule is done for all w vectors and we can find the w values that will minimize he loss function

## Some important functions that we have:
- ```get_predictions(A2):``` this is a function meant to basically be the function predictor that deternmines from the output layer which is the most likely number we can classify

- ```initialize_parameters():``` this is how we initialize all of the weight vectors and the biases that we can use, this is what we initially call gradient descent on which will change these values into the 
w and b values hat minimize the loss function

- ```get_accuracy(predictions, Y):``` this is a direct comparison between the outputted label and the actual value of the image (y).

- ```one_hot(Y):``` This allows us to chnage the y label to being a vector with size 10. Considering the output of our neural network is a vector of probabilities we can use the one_hot encoded y 
value as comparing the vector after the function gives the max probability inside of the outputted vector from the neural network.
ex) if we have probabilities (0.1, 0.05, 0.05, 0.2, 0.1, 0.02, 0.03, 0.02, 0.03, 0.40) then the get_predcitions will output (0,0,0,0,0,0,0,0,0,1)
and if correct label is 9 then the one hot encoded vector will be (0,0,0,0,0,0,0,0,1) which matches so this is a correct prediction.


