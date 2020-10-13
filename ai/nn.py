# -------------------------------------------------------------------------
import torch
import pickle
import os
from Players.minimax import MiniMaxPlayer
from Players.mcts import MCTSPlayer
from Players.memory import MemoryDict
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
# -------------------------------------------------------------------------

N, D_in, H, D_out = 64, 10, 5, 1

# x = torch.randn(N, D_in)
# x = MemoryDict.fill_mem()
y = torch.randn(N, D_out)

filename = Path("Players/Memory/MCTree.p")


with open(filename, "rb") as f:
    x += f
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(x, encoding="utf-8")

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4

for t in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate + param.grad
