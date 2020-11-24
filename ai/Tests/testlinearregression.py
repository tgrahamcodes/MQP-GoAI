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

#-------------------------------------------------------------------------
def test_python_version():
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
def test_neural_net():  
    # Hardcoded numpy arrays to test model "normalized"
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
    print(model.train(inputs, labels))

    ground_truth = Variable(torch.Tensor(([12])))
    pred_y = model(ground_truth)
    print("predict", 25, int(model(ground_truth).item()))

    assert (int(model(ground_truth).item()) == 25 or int(model(ground_truth).item()) == 24)
    assert (model)
    assert (type(model) == LinearRegression)
    assert (model.lin.in_features == 1)
    assert (model.lin.out_features == 1)
#-------------------------------------------------------------------------

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

    assert (model)
    assert (type(model) == LinearRegression)
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

    # Instatiating the model
    model = LinearRegression(1, 1)  
    model.file =  'Players/Memory/MM_LinearReg.p'
    model.save('Players/Memory/MM_LinearReg.p')

    newModel = LinearRegression(1,1)
    newModel.file = 'Players/Memory/MM_LinearReg.p'
    newModel.load('Players/Memory/MM_LinearReg.p')
    
    assert (model)
    assert (newModel)
    assert (type(newModel) == LinearRegression)
    assert (type(model) == LinearRegression)
    assert (path.exists('Players/Memory/MM_LinearReg.p'))
    assert Path(__file__).parents[1].joinpath('Players/Memory/MM_LinearReg.p')
#-------------------------------------------------------------------------