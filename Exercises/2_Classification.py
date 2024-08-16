import torch
from torch import nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy

device = "cuda"
torch.set_default_device(device)
x, y = make_moons(n_samples=1000, noise=0.03, random_state=42)
x = torch.from_numpy(x).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
def acc_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # eq runs when tensors are =, so measures # of correct guesses
    acc = (correct/len(y_pred)) * 100
    return acc
class moon(nn.Module):
    def __init__(self,in_layers,out_layers,hidden_layers):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_layers,hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers,hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers,out_layers)
        )
    def forward(self,x):
            return self.layers(x)
moon_model = moon(in_layers=2,out_layers=1,hidden_layers=8).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=moon_model.parameters(),lr=0.1)
# training time
torch.manual_seed(42)
epochs = 1000
x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)
for epoch in range(epochs):
    moon_model.train()
    logits = moon_model(x_train).squeeze()
    preds = torch.round(torch.sigmoid(logits))
    acc = acc_fn(y_train,preds)
    loss = loss_fn(logits,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    moon_model.eval()
    with torch.inference_mode():
        test_logits = moon_model(x_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))
        test_acc = acc_fn(y_test,test_preds)
        test_loss = loss_fn(test_logits,y_test)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Train Loss: {loss} | Train Accuracy: {acc} | Test Loss: {test_loss} | Test Accuracy: {test_acc}")