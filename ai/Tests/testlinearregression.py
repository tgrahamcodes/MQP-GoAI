import numpy as np
import sys
import torch.nn as nn
import torch
import math
from ..Players.linearregression import LinearRegression
import os.path
from os import path
from pathlib import Path
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
#-------------------------------------------------------------------------
def test_python_version():
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)
#-------------------------------------------------------------------------

# #-------------------------------------------------------------------------
# def test_neural_net():  
#     x_data = [i for i in range(11)]
#     y_data = [2*i + 1 for i in x_data]

#     x_train = np.array(x_data, dtype=np.float32)
#     x_train.shape

#     y_train = np.array(y_data, dtype=np.float32)
#     y_train.shape

#     x_train = x_train.reshape(-1, 1)
#     x_train.shape

#     y_train = y_train.reshape(-1, 1)
#     y_train.shape

#     inputs = torch.from_numpy(x_train).requires_grad_()
#     labels = torch.from_numpy(y_train)

#     # Instatiating the model
#     model = LinearRegression(1, 1)
#     print(model.train(inputs, labels))
# #-------------------------------------------------------------------------

#-------------------------------------------------------------------------
def test_data_save():
    x_data = [i for i in range(11)]
    y_data = [2*i + 1 for i in x_data]

    x_train = np.array(x_data, dtype=np.float32)
    x_train.shape

    y_train = np.array(y_data, dtype=np.float32)
    y_train.shape

    x_train = x_train.reshape(-1, 1)
    x_train.shape

    y_train = y_train.reshape(-1, 1)
    y_train.shape

    inputs = torch.from_numpy(x_train).requires_grad_()
    labels = torch.from_numpy(y_train)

    # Instatiating the model
    model = LinearRegression(1, 1)
    model.file = 'Players/Memory/MM_LinearReg.p'
    model.save('Players/Memory/MM_LinearReg.p')

    # Asserting the model is non null
    assert (model and model.lin)

    # Asserting the type of the model is correct
    assert (type(model) == LinearRegression)

    # Asserting the path is correct for the file to be saved
    assert (path.exists('Players/Memory/MM_LinearReg.p'))
    assert Path(__file__).parents[1].joinpath('Players/Memory/MM_LinearReg.p')
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
def test_data_load():
    # Instatiating the model
    model = LinearRegression(1,1)
    model.load('Players/Memory/MM_LinearReg.p')
    model.file = 'Players/Memory/MM_LinearReg.p'
    assert (path.exists('Players/Memory/MM_LinearReg.p'))
    assert (model.lin.state_dict)
    assert (model.lin.weight.data)
    assert (model.lin.bias)
    assert (model.parameters)
    assert (model.lr == 0.0001)
    assert (model.epochs == 500)
    assert (model)
    assert (type(model) == LinearRegression)
    assert (model.lin.in_features == 1)
    assert (model.lin.out_features == 1)
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
def test_data_save_and_load():
    x_data = [1,2,3]
    y_data = [2*i + 1 for i in x_data]

    x_train = np.array(x_data, dtype=np.float32)
    x_train.shape

    y_train = np.array(y_data, dtype=np.float32)
    y_train.shape

    inputs = torch.from_numpy(x_train).requires_grad_()
    labels = torch.from_numpy(y_train)

    # Instatiating the model and saving it
    model = LinearRegression(1, 1)
    model.file =  'Players/Memory/MM_LinearReg.p'
    model.save('Players/Memory/MM_LinearReg.p')

    # Loading the model that was saved
    newModel = LinearRegression(1,1)
    newModel.file = 'Players/Memory/MM_LinearReg.p'
    newModel.load('Players/Memory/MM_LinearReg.p')

    newerModel = LinearRegression(2,2)
    newerModel.lin.weight = torch.nn.Parameter(torch.Tensor(4))
    newerModel.save('Players/Memory/MM_LinearReg2.p')
    
   # print("Model's state_dict:")
   # for param_tensor in newerModel.lin.state_dict():
   #     print(param_tensor, "\t", newerModel.lin.state_dict()[param_tensor].size())
  
    # Asserting the model has the same features as the loaded model
    assert (torch.allclose(model.lin.bias, newModel.lin.bias))
    assert (torch.allclose(model.lin.weight, newModel.lin.weight))

    assert (len(newerModel.lin.weight.data) == 4)
    assert (len(newerModel.lin.bias) == 2)

    # Asserting the objects are not null
    assert (model and model.lin)
    assert (newModel and newModel.lin)

    # Asserting the type of the object is correct
    assert (type(newModel) == LinearRegression)
    assert (type(model) == LinearRegression)

    # Asserting the path exists for the Memory file
    assert (path.exists('Players/Memory/MM_LinearReg.p'))
    assert Path(__file__).parents[1].joinpath('Players/Memory/MM_LinearReg.p')
#-------------------------------------------------------------------------

def test_neural_net():

    x_data = [1,2,3]
    y_data = [2*i + 1 for i in x_data]

    x_train = np.array(x_data, dtype=np.float32)
    x_train.shape

    y_train = np.array(y_data, dtype=np.float32)
    y_train.shape

    x = torch.from_numpy(x_train).requires_grad_()
    y = torch.from_numpy(y_train)

    model = LinearRegression(1, 1)
    model.file = 'Players/Memory/MM_LinearReg.p'

    class sample_data(Dataset):
        def __init__(self, x, y):
            self.x = x
            self.y = y
        def __len__(self):
            return len(self.x) 
        def __getitem__(self, index):
            x = self.x[index]
            y = self.y[index]
            return x, y
    d = sample_data(x, y)
    data_loader = DataLoader(d, batch_size=1, shuffle=False, num_workers=0)
    model.train(data_loader)

    ground_truth = Variable(torch.Tensor(([12])))
    pred_y = model(ground_truth)
    print("predict", 25, int(model(ground_truth).item()))
    
    # Asserting that the prediction is correct
    assert (math.isclose(25, model(ground_truth).item(), abs_tol=1.5))

    # Asserting the model is non null
    assert (model and model.lin)

    # Asserting the type of the model is correct
    assert (type(model) == LinearRegression)

    # Asserting the in and out features are correct
    assert (model.lin.in_features == 1)
    assert (model.lin.out_features == 1)
    #-------------------------------------------------------------------------
