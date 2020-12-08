from problem1 import *
import sys
import math
from game import TicTacToe
import numpy as np
from torch.utils.data import Dataset, DataLoader
'''
    Unit test 1:
    This file includes unit tests for problem1.py.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 1 (15 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3.6 or above 
    assert sys.version_info[1]>=6

#---------------------------------------------------
def test_compute_z():
    ''' (5 points) compute_z'''
    g = TicTacToe()
    m = PNet()
    x= g.s.unsqueeze(0)
    z = compute_z(x,m)
    assert type(z) == th.Tensor 
    assert z.size()[0] == 1 
    assert z.size()[1] == 9 
    assert z.requires_grad
#---------------------------------------------------
def test_compute_L():
    ''' (5 points) compute_L'''
    z = th.tensor([[ 0.1,-0.2, 0.0], # linear logits for the first sample in the mini-batch
                   [ 0.0,-0.1, 0.1], # linear logits for the second sample in the mini-batch
                   [-0.1, 0.0, 0.2], # linear logits for the third sample in the mini-batch
                   [-0.2, 0.1, 0.3]], requires_grad=True) # linear logits for the last sample in the mini-batch
    # the labels of the mini-batch: vector of length 4 (batch_size)
    y = th.LongTensor([1,2,1,0])
    L = compute_L(z,y)
    assert type(L) == th.Tensor 
    assert L.requires_grad
    assert np.allclose(L.detach().numpy(),1.2002,atol=1e-4) 
    L.backward() # back propagate gradient to W and b
    dL_dz_true = [[ 0.0945, -0.1800,  0.0855],
                  [ 0.0831,  0.0752, -0.1582],
                  [ 0.0724, -0.1700,  0.0977],
                  [-0.1875,  0.0844,  0.1031]]
    assert np.allclose(z.grad,dL_dz_true, atol=0.01)
    z = th.tensor([[  0.1,-1000, 1000], # linear logits for the first sample in the mini-batch
                   [  0.0, 1100, 1000], # linear logits for the second sample in the mini-batch
                   [-2000,-1900,-5000]], requires_grad=True) # linear logits for the last sample in the mini-batch
    y = th.LongTensor([2,1,1])
    L = compute_L(z,y)
    assert np.allclose(L.data,0,atol=1e-4) 
#---------------------------------------------------
def test_train():
    ''' (5 points) train'''
    X = th.Tensor([[[[0., 0., 0.],
                     [0., 0., 0.],
                     [0., 0., 0.]],
                    [[0., 0., 0.],
                     [0., 0., 0.],
                     [0., 0., 0.]],
                    [[1., 1., 1.],
                     [1., 1., 1.],
                     [1., 1., 1.]]],
                   [[[0., 0., 0.],
                     [0., 0., 0.],
                     [0., 0., 0.]],
                    [[0., 0., 0.],
                     [0., 0., 0.],
                     [0., 0., 0.]],
                    [[1., 1., 1.],
                     [1., 1., 1.],
                     [1., 1., 1.]]] ])
    Y= th.LongTensor([4,5])
    class toy_dataset(Dataset):
        def __init__(self):
            self.X  = X 
            self.Y = Y
        def __len__(self):
            return 2 
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    d=toy_dataset()
    data_loader = DataLoader(d, batch_size=2, shuffle=True, num_workers=0)
    m = train(data_loader, alpha=0.1, n_epoch=150)
    l = th.nn.Softmax(dim=1)
    z = m(X)
    p = l(z)
    np.allclose(p[:,4:6].data,0.5*np.ones((2,2)),atol = 0.05)

