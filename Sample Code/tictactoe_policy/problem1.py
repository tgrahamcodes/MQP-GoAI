import torch as th
from model import PNet
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 1: Policy Network with Supervised Training (15 points)
    In this problem, you will implement neural network for multi-class classification problems.
This neural network can be used as the policy network for TicTacToe game. It takes a game state (a tensor of 3 input channels, 3 X 3 board size) as the input and use a convolutional neural network to compute the probability of each action.

    A list of all variables being used in this problem is provided at the end of this file.
'''

#----------------------------------------------------
'''
    Given a policy network model (m) as defined in 'model.py', please compute the linear logits z on a mini-batch of data samples x1, x2, ... x_batch_size. In the mean time, please also connect the global gradients of the linear logits z (dL_dz) with the global gradients of the weights dL_dW and the biases dL_db in the PyTorch tensors. 
    ---- Inputs: --------
        * x: the feature vectors of a mini-batch of data samples, a float torch tensor of shape (batch_size, p).
        * m: .
    ---- Outputs: --------
        * z: the linear logits on a mini-batch of data samples, a float torch tensor of shape (batch_size, c).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_z(x, m):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    z = m(x)
    #########################################
    return z
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py:test_compute_z
        --- OR ---- 
        python3 -m nose -v test1.py:test_compute_z
        --- OR ---- 
        python -m nose -v test1.py:test_compute_z
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Multi-class Cross Entropy Loss) Suppose we are given a model and we have already computed the linear logits z on a mini-batch of training samples. Suppose the labels of the training samples are in y. Please compute the average loss of the softmax regression model on the mini-batch of training samples. In the mean time, please also connect the global gradients of the linear logits z (dL_dz) with the loss L correctly. 
    ---- Inputs: --------
        * z: the linear logits on a mini-batch of data samples, a float torch tensor of shape (batch_size, c).
        * y: the labels of a mini-batch of data samples, a torch integer vector of length batch_size. The value of each element can be 0,1,2, ..., or (c-1).
    ---- Outputs: --------
        * L: the average multi-class cross entropy loss on a mini-batch of training samples, a torch float scalar.
    ---- Hints: --------
        * The loss L is a scalar, computed from the average of the cross entropy loss on all samples in the mini-batch. For example, if the loss on the four training samples are 0.1, 0.2, 0.3, 0.4, then the final loss L is the average of these numbers as (0.1+0.2+0.3+0.4)/4 = 0.25. 
        * You could use CrossEntropyLoss in PyTorch to compute the loss. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_L(z, y):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    L = th.nn.CrossEntropyLoss()(z,y)
    #########################################
    return L
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py:test_compute_L
        --- OR ---- 
        python3 -m nose -v test1.py:test_compute_L
        --- OR ---- 
        python -m nose -v test1.py:test_compute_L
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Training Policy Network) Given a training dataset X (game states), Y (labels) in a data loader, train a policy network model using mini-batch stochastic gradient descent: iteratively update the parameters using the gradients on each mini-batch of data samples.  We repeat n_epoch passes over all the training samples. 
    ---- Inputs: --------
        * data_loader: the PyTorch loader of a dataset.
        * alpha: the step-size parameter of gradient descent, a float scalar.
        * n_epoch: the number of passes to go through the training dataset in the training process, an integer scalar.
    ---- Outputs: --------
        * m: .
    ---- Hints: --------
        * Step 1 Forward pass: compute the linear logits z and loss L. 
        * Step 2 Back propagation: compute the gradients of parameters. 
        * Step 3 Gradient descent: update the parameters using gradient descent. 
        * This problem can be solved using 4 line(s) of code.
'''
#---------------------
def train(data_loader, alpha=0.001, n_epoch=100):
    m = PNet() # initialize the model
    optimizer = th.optim.SGD(m.parameters(), lr=alpha) # create an SGD optimizer
    for _ in range(n_epoch): # iterate through the dataset n_epoch times
        for mini_batch in data_loader: # iterate through the dataset, with one mini-batch of random training samples (x,y) at a time
            x=mini_batch[0] # the game states in a mini-batch
            y=mini_batch[1] # the actions of the game states in a mini-batch
            #########################################
            ## INSERT YOUR CODE HERE (5 points)
            L = compute_L(m(x),y)
            L.backward()
            optimizer.step()
            optimizer.zero_grad()
            #########################################
    return m
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py:test_train
        --- OR ---- 
        python3 -m nose -v test1.py:test_train
        --- OR ---- 
        python -m nose -v test1.py:test_train
        ---------------------------------------------------
    '''
    
    


#--------------------------------------------

''' 
    TEST problem 1: 
        Now you can test the correctness of all the above functions by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py
        --- OR ---- 
        python3 -m nose -v test1.py
        --- OR ---- 
        python -m nose -v test1.py
        ---------------------------------------------------

        If your code passed all the tests, you will see the following message in the terminal:
        ----------- Problem 1 (15 points in total)--------------------- ... ok
        * (5 points) compute_z ... ok
        * (5 points) compute_L ... ok
        * (5 points) train ... ok
        ----------------------------------------------------------------------
        Ran 3 tests in 1.489s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* n:  the number of data instance in the training set. 
* p:  the number of input features. 
* c:  the number of classes in the classification task, an integer scalar. 
* batch_size:  the number of samples in a mini-batch, an integer scalar. 
* x:  the feature vectors of a mini-batch of data samples, a float torch tensor of shape (batch_size, p). 
* y:  the labels of a mini-batch of data samples, a torch integer vector of length batch_size. The value of each element can be 0,1,2, ..., or (c-1). 
* W:  the weight matrix of softmax regression, a float torch Tensor of shape (p by c). 
* b:  the bias values of softmax regression, a float torch vector of length c. 
* z:  the linear logits on a mini-batch of data samples, a float torch tensor of shape (batch_size, c). 
* a:  the softmax activations on a mini-batch of data samples, a float torch tensor of shape (batch_size, c). 
* L:  the average multi-class cross entropy loss on a mini-batch of training samples, a torch float scalar. 
* data_loader:  the PyTorch loader of a dataset. 
* alpha:  the step-size parameter of gradient descent, a float scalar. 
* n_epoch:  the number of passes to go through the training dataset in the training process, an integer scalar. 
* y_predict:  the predicted labels of a mini-batch of test data samples, a torch integer vector of length batch_size. y_predict[i] represents the predicted label on the i-th test sample in the mini-batch. 
* optimizer:  a PyTorch optimizer (such as SGD, ADAM, RMSProp) to handle the gradient descent for parameters in the model (W and b). 

'''
#--------------------------------------------
