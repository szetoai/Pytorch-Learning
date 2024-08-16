from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

# 1. make data
samples = 1000
x, y = make_circles(samples, noise=0.03, random_state=42) # x is two layers, y is labels (either 0 or 1)
# Turn data into tensors and create train/test splits
x = torch.from_numpy(x).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# 2. build model
device = "cuda" if torch.cuda.is_available() else "cpu"
class circle(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2,out_features=10)
        self.layer2 = nn.Linear(in_features=10,out_features=10)
        self.layer3 = nn.Linear(in_features=10,out_features=1)
        self.relu = nn.ReLU() # needed for non linear data
    def forward(self, x):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))
circle_model = circle().to(device)
# 2.1 Loss and Optimizer
loss_func = nn.BCEWithLogitsLoss() # can use for binary classification
optimizer = torch.optim.SGD(params=circle_model.parameters(),lr=0.1)
# new! accuracy function or evaluation metric
def accuracy_func(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # eq runs when tensors are =, so measures # of correct guesses
    acc = (correct/len(y_pred)) * 100
    return acc
# 3. train model
y_logits = circle_model(x_test.to(device))[:5]
y_pred_probs = torch.sigmoid(y_logits) # turn logits into understandable form
y_preds = torch.round(y_pred_probs) # round probs into either 0 or 1 to compare
y_preds = y_preds.squeeze() # make it 1d to match original labels
print(y_preds)
# all the same stuff
torch.manual_seed(42)
epochs = 1000
x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)
for epoch in range(epochs):
    circle_model.train() # forward
    y_logits = circle_model(x_train).squeeze() # logits are pure output from formula
    y_pred = torch.round(torch.sigmoid(y_logits)) # round the sigmoid (the understandable version of logits only for binary) to either 0 or 1 (binary)
    loss = loss_func(y_logits, y_train) # loss
    acc = accuracy_func(y_true=y_train,y_pred=y_pred)
    optimizer.zero_grad() # zero grad
    loss.backward() # backprop
    optimizer.step() # step
    # test time
    circle_model.eval()
    with torch.inference_mode():
        test_logits = circle_model(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_func(y_test,test_logits)
        test_acc = accuracy_func(y_true=y_test,y_pred=test_pred)
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train Loss: {loss} | Train Accuracy: {acc} | Test Loss: {test_loss} | Test Accuracy: {test_acc}")