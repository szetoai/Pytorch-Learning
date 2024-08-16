import torch
from torch import nn
import matplotlib.pyplot as plt

# original data + formula
weight = 0.3
bias = 0.9
input = torch.arange(80,120,0.25) # og data
output = weight * input + bias # formula
split_percent = int(0.8*len(input)) # splitting data into train/test
xtrain, ytrain = input[:split_percent], output[:split_percent]
xtest, ytest = input[split_percent:], output[split_percent:]
# plotting function
def plot(train_x = xtrain, train_y = ytrain, test_x = xtest, test_y = ytest, predictions = None):
    plt.figure(figsize=(10,10)) # create a window
    plt.scatter(train_x, train_y, c="g", s=10, label="OG Training Data") # x, y, color, size, label
    plt.scatter(test_x, test_y, c="r", s=10, label="OG Testing Data")
    if predictions != None: # plot the predictions if they exist
        plt.scatter(test_x,predictions,c="b",s=10,label="AI Test")
    plt.legend(prop={"size":10})
    plt.show()
# create linreg type for models
class LinReg(nn.Module): # nn.Module is parent
    def __init__(self): # initialize class
        super().__init__() # yoink from parent
        self.weight = nn.Parameter(torch.randn(1,dtype=torch.float),requires_grad=True) # create a random weight to start
        self.bias = nn.Parameter(torch.randn(1,dtype=torch.float),requires_grad=True) # same but with bias
    def forward(self, x): # return the result of the random formula
        return self.weight * x + self.bias
# creating model
review_model = LinReg()
# preview (not necessary but is good for comparison)
with torch.inference_mode():
    preds = review_model(xtest)
plot(predictions=preds)
# training arc
loss_func = nn.L1Loss() # prebuilt loss func; remember loss is the amount away from real
optimizer_func = torch.optim.SGD(params=review_model.parameters(),lr=0.0001) # inherit review_model's parameters and set learning rate per step to low
train_loss_values, test_loss_values, epoch_count = [], [], [] # to track progress while learning
epochs = 2500
# learning start! 
'''
0. set model to train mode
1. forward (use original formula from model to predict)
2. loss (find difference between predicted and real)
3. zero gradient (set gradient of optimizer to 0 for adjustment)
4. backpropagation (compute gradient of loss to update parameters)
5. step (update optimizer to improve parameters)
'''
for i in range(epochs):
    review_model.train() # train mode
    epoc_pred = review_model(xtrain) # forward
    loss = loss_func(epoc_pred, ytrain) # loss
    optimizer_func.zero_grad() # zero grad
    loss.backward() # backprop
    optimizer_func.step() # step
    # training done, time to test!
    review_model.eval() # test mode
    with torch.inference_mode():
        test_pred = review_model(xtest) # final step
        test_loss = loss_func(test_pred,ytest) # final loss
        # showing progress (semi-optional)
        if i % 25 == 0: # every 25 epochs
            epoch_count.append(i)
            train_loss_values.append(loss.detach().numpy()) # the MAE loss value for this epoch
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {i} | Train Loss: {loss} | Test Loss: {test_loss}")
# prediction time!
review_model.eval()
with torch.inference_mode():
    preds = review_model(xtest)
plot(predictions=preds)
print(f"AI weight and bias: {review_model.state_dict()}")
# export the model
torch.save(obj=review_model.state_dict(),f="Models/My_Second_Model") # save the parameters, to this filepath
model_copy = LinReg()
# load the model
model_copy.load_state_dict(torch.load(f="Models/My_Second_Model",weights_only=True))
with torch.inference_mode():
    preds = model_copy(xtest) + -2
plot(predictions=preds)