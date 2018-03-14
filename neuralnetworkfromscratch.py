# Creating a neural network that can simulate an OR GATE from scratch
from numpy import array,random,dot,exp,sum

def sigmoid(x):
    return 1/(1+exp(x))

def sigmoid_derivative(x):
    return x*(1-x)

def Neural_Networks():
    inputs=array([[0,0,0],[0,1,0],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
    expected_outputs=array([[0],[1],[1],[1],[1],[1]])
    predict=array([0,0,1])
    weights=random.random((3,1))
    updated_weights=train(inputs,expected_outputs,weights,1000)
    print(predict_output(predict,updated_weights))
    return

def train(inputs,outputs,weights,iterations):

    for i in xrange(iterations):
        output=sigmoid(dot(inputs,weights))
        loss=outputs-output
        #print(sum(loss))

        adjustment=dot(inputs.T,loss*sigmoid_derivative(output))

        weights=weights+adjustment
    return weights

def predict_output(predict,weights):
    return round(sigmoid(dot(predict,weights)))

Neural_Networks()
