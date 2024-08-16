# Import dependencies
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
# create data
classes = 4
features = 2
seed = 42
x_blob, y_blob = make_blobs(n_samples=1000, n_features=features, centers=classes, cluster_std=1.5, random_state=seed)
# turn into tensors
x_blob = torch.from_numpy(x_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)
# split
x_train, x_test, y_train, y_test = train_test_split(x_blob, y_blob, test_size=0.2, random_state=seed)
# model
class blob(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )
    def forward(self, x):
        return self.stack(x)
def accuracy_func(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # eq runs when tensors are =, so measures # of correct guesses
    acc = (correct/len(y_pred)) * 100
    return acc
blob_model = blob(input_features=features,output_features=classes,hidden_units=8).to(device)
# loss and optim
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(blob_model.parameters(),lr=0.1)
# training arc!
torch.manual_seed(42)
epochs = 1000
x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)
for epoch in range(epochs):
    blob_model.train() # train mode
    logits = blob_model(x_train) # raw output/forward
    preds = torch.softmax(logits, dim=1).argmax(dim=1) # understandable probabilities and labels
    loss = loss_func(logits,y_train) # loss
    acc = accuracy_func(y_true=y_train,y_pred=preds)
    optimizer.zero_grad() # zero grad
    loss.backward()
    optimizer.step()
    # testing
    blob_model.eval()
    with torch.inference_mode():
        test_logits = blob_model(x_test)
        test_preds = torch.softmax(test_logits,dim=1).argmax(dim=1)
        test_loss = loss_func(test_logits,y_test)
        test_acc = accuracy_func(y_true=y_test,y_pred=test_preds)
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train Loss: {loss} | Train Accuracy: {acc} | Test Loss: {test_loss} | Test Accuracy: {test_acc}")
# Turn predicted logits in prediction probabilities
y_pred_probs = torch.softmax(test_logits, dim=1)

# Turn prediction probabilities into prediction labels
y_preds = y_pred_probs.argmax(dim=1)
from torchmetrics import Accuracy
# Compare first 10 model preds and test labels
final_acc = Accuracy(task="multiclass",num_classes=4)
print(final_acc(y_preds,y_test))