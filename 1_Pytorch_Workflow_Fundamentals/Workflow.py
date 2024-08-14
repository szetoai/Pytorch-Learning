import torch
from torch import nn
import matplotlib.pyplot as plt

# 1.1 - Data
# create known parameters
weight = 0.7
bias = 0.3
# create data
start = 0
end = 1
step = 0.02
x = torch.arange(start, end, step)
y = weight * x + bias
# creat training/tesing split
train_split = int(0.8*len(x))
x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]
# plotting data
def plot_predictions(train_data=x_train, 
                     train_labels=y_train, 
                     test_data=x_test, 
                     test_labels=y_test, 
                     predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data") # x, y, color, scale, label
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    # showing legend
    plt.legend(prop={"size":14})
    plt.show()

# 1.2 - Build Model 
# Create a Linear Regression model class
class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(torch.randn(1, # <- start with random weights (this will get adjusted as the model learns)
                                                dtype=torch.float), # <- PyTorch loves float32 by default
                                   requires_grad=True) # <- can we update this value with gradient descent?)

        self.bias = nn.Parameter(torch.randn(1, # <- start with random bias (this will get adjusted as the model learns)
                                            dtype=torch.float), # <- PyTorch loves float32 by default
                                requires_grad=True) # <- can we update this value with gradient descent?))

    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data (e.g. training/testing features)
        return self.weights * x + self.bias # <- this is the linear regression formula (y = m*x + b)
    
model = LinearRegressionModel()
with torch.inference_mode():
    y_preds = model(x_test)
plot_predictions(predictions=y_preds)

# 1.3 Train model
loss_fn = nn.L1Loss() # Mean Absolute Error is L1Loss
optimizer = torch.optim.SGD(params=model.parameters(),lr=0.01) # SGD = Stochastic gradient descent
# Set the number of epochs (how many times the model will pass over the training data)
epochs = 2500
# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []
for epoch in range(epochs):
    # TRAINING
    model.train() # training mode
    # 1. Forward pass on x train data (does math stuff to initial data) to get prediction
    y_pred = model(x_train)
    # 2. Calculate loss (difference between prediction and real)
    loss = loss_fn(y_pred, y_train)
    # 3. Set gradient to zero
    optimizer.zero_grad()
    # 4. Find gradient of the loss function using backpropagation
    loss.backward()
    # 5. Edit optimizer
    optimizer.step()
    # TESTING
    model.eval() # evaluation mode
    with torch.inference_mode():
        # 1. Forward pass on test x data (does math stuff to data)
        test_pred = model(x_test)
        # 2. Calculate loss for test data
        test_loss = loss_fn(test_pred, y_test.type(torch.float))
        # printing out results every 10 epochs
        if epoch % 100 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()
print("The model learned the following values for weights and bias:")
print(model.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")

# 1.4 Making predictions with trained model
# 1. evaluation mode again
model.eval()
# 2. inference mode
with torch.inference_mode():
    y_preds = model(x_test)
plot_predictions(predictions=y_preds)

# 1.5 Saving and loading model
from pathlib import Path
# 1. create models directory
model_path = Path("Models")
model_path.mkdir(parents=True, exist_ok=True)
# 2. create models save path
model_name = "My_First_Model"
model_save_path = model_path/model_name
# 3. save the model state dict (parameters/function stuff)
print(f"Saving to: {model_save_path}")
torch.save(obj=model.state_dict(), f=model_save_path) # saves learned parameters
# 4. loading the a new model with same parameters
model2 = LinearRegressionModel()
model2.load_state_dict(torch.load(f=model_save_path, weights_only=True))