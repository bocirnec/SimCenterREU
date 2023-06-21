"""
A code to apply CNN for time series prediction of top floor acceleration.
"""

#Load the required libraries
import torch 
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.optim as optim
from sklearn.metrics import accuracy_score
import argparse
import time

#-----------------------------------------------------------------------------
#Prepare the data 
#-----------------------------------------------------------------------------
# dt = 1.0
dt = 0.2

#Read the data and use only the top floor acceleration for the prediction 
y = torch.tensor(np.loadtxt("Data/acc1"), dtype=torch.float32)

Nt = np.shape(y)[0]
x = torch.tensor(np.arange(0.0, Nt*dt, dt), dtype=torch.float32)

plt.figure(figsize = (12,4))
plt.xlim(-10, Nt*dt + 10)
plt.grid(True)
plt.plot(x, y)

batch_size = 62
test_size = 1000
val_size = 1000 
pred_window_size = 10 #Prediction window size
train_size = Nt - (val_size + test_size)


train_dataset = y[:train_size]
val_dataset =  y[train_size : train_size + val_size]
test_dataset = y[-test_size:]

print("Train set: {}, Validation set: {}, Test set: {}".format(train_size, val_size, test_size))

#-----------------------------------------------------------------------------
#Intersect data set 
#-----------------------------------------------------------------------------
# examples = enumerate(train_loader)

# print(len(train_dataset), len(val_dataset), len(test_dataset))

# batch_idx, (example_data, example_targets) = next(examples) 

plt.figure(figsize = (12,4))
plt.xlim(-10, dt*Nt+1)
plt.grid(True)
plt.plot(x[:train_size], train_dataset, 'k-')
plt.plot(x[train_size : train_size + val_size], val_dataset, 'b-')
plt.plot(x[-test_size:], test_dataset, 'r-')


#-----------------------------------------------------------------------------
#Model defination 
#-----------------------------------------------------------------------------
def prepareInputData(data, bs, ps):
  batch_data = []
  L = len(data)
  
  nbatchs  = int((L-ps)/bs) 
  
  print("Length of input training data = {}".format(L))
  for i in range(nbatchs):
      start  = i*bs
      end = (i+1)*bs
      window = data[start : end]      
      label = data[end : end + ps]
      batch_data.append((window, label))   
      
  return batch_data

window_size = batch_size
train_data = prepareInputData(train_dataset, batch_size, pred_window_size)
val_data = prepareInputData(val_dataset, batch_size, pred_window_size)
test_data = prepareInputData(test_dataset, batch_size, pred_window_size)

#Show sample train data
plt.stem(x[:window_size].numpy(), train_data[0][0].numpy())
plt.stem(x[batch_size:batch_size+pred_window_size].numpy(), train_data[0][1].numpy(), 'r-')
print(train_data[0][1])

##############################################################################
#Linear Model
##############################################################################
class LinearModel(nn.Module):
    def __init__(self, in_dim, out_dim): 
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
    
    def forward(self, x):
        x = self.linear(x)
        return x

##############################################################################
#CNN Model
##############################################################################
class MyCNN(nn.Module):
    def __init__(self, 
                 kernel_size=5, 
                 stride_size=1, 
                 num_channel=1,
                 depth_1=1,
                 depth_2=5,
                 kernel_size2=5,                 
                 num_hidden=20, 
                 num_labels=5
                 ): 
        
        super(MyCNN, self).__init__()
        # self.in_dim = in_dim
        # self.out_dim = out_dim
        # self.hid_dim = hid_dim
        # self.n_layers = n_layers
        # self.act = act
        # self.dropout = dropout
        # self.use_bn = use_bn
        # self.use_xavier = use_xavier
        
        self.classifier = nn.Sequential
        (
            nn.Conv1d
            (
                num_channel, 
                depth_1, 
                kernel_size=kernel_size
             )
        )
        
        self.fc1 = nn.Sequential
        (
            nn.Linear(depth_2*kernel_size2, num_hidden),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.fc2 = nn.Sequential
        (
            nn.Linear(num_hidden, num_labels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
    def forward(self, x):
        x = self.classifier(x)
        
        return x
        


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
        
        
X=train_dataset

y = torch.Tensor(train_size)

for i in range(len(y)-1):
    y[i] = X[i+1]



#print(y)
batch_size=5

model=MyCNN()
optimizer=optim.Adam(model.parameters(), lr=0.01) #l2
criterion=nn.CrossEntropyLoss()

model.train()

values=[]
#print(X[:32])
for epoch in range(10):
    for batch in iterate_minibatches(X[:len(y)], y, batch_size):
        x_sublist,y_sublist=batch
        X2_tensor= torch.Tensor(np.array(x_sublist)).reshape(1,1,-1) #3 dimensional
        y2_tensor= torch.Tensor(np.array(y_sublist)).reshape(1,1,-1) #3 dimensional
        outputs=model(X2_tensor)
        print(X2_tensor,outputs)
        #loss_value=torch.mean((outputs - y2_tensor)**2)
        #print(loss_value)
        #loss=criterion(outputs,targets)
        #loss_value.backward()
        #optimizer2.step()
        #print('[%d] loss: %.3f' % (epoch, loss_value.item()))        
    
print(outputs)

#         self.fc = nn.Linear(self.in_dim, self.hid_dim)

#         self.linears = nn.ModuleList()
#         self.bns = nn.ModuleList()
#         for i in range(self.n_layers-1):
#             self.linears.append(nn.Linear(self.hid_dim, self.hid_dim))
#             if self.use_bn:
#                 self.bns.append(nn.BatchNorm1d(self.hid_dim))
#         self.fc2 = nn.Linear(self.hid_dim, self.out_dim)

#         if self.act == 'relu':
#             self.act = nn.ReLU()
        
#         self.dropout = nn.Dropout(self.dropout)
#         if self.use_xavier:
#             for linear in self.linears:
#                 nn.init.xavier_normal_(linear.weight)
#                 linear.bias.data.fill_(0.01)
    
#     def forward(self, x):
#         x = self.act(self.fc(x))
#         for i in range(len(self.linears)):
#             x = self.act(self.linears[i](x))
#             x = self.bns[i](x)
#             x = self.dropout(x)
#         x = self.fc2(x)
#         return x

# #------------------------------------------------------------------------------
# # CNN Model
# #------------------------------------------------------------------------------
# def trainModel(args):
#     #model = LinearModel(args.in_dim, args.out_dim)

#     model = MLPModel(args.in_dim, args.out_dim, args.hid_dim, args.n_layers, args.act, args.dropout, args.use_bn, args.use_xavier)

#     print(model)

#     # ====== Loss function ====== #
#     #criterion = nn.MSELoss()
#     criterion = nn.CrossEntropyLoss() #Image classification

#     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

#     # ====== Data collection ====== #
#     list_epoch = [] 
#     list_train_loss = []
#     list_val_loss = []
#     list_acc = []
#     list_acc_epoch = []

#     # ====== Loop ====== #
#     for epoch in range(args.epoch):  
        
#         # ====== Train ====== #
#         model.train() # Set the model be 'train mode' 
#         train_loss = 0 # to sum up each batch
        
#         for input_X, true_y in train_data:
#             optimizer.zero_grad() # Initialize the gradient in the optimizer

#             input_X = input_X.squeeze()
#             input_X = input_X.view(-1, batch_size)

#             pred_y = model(input_X)

#             loss = criterion(pred_y.squeeze(), true_y)
#             loss.backward()
#             optimizer.step()
#             train_loss +=loss.item()

#         train_loss = train_loss / len(train_data)
#         list_train_loss.append(train_loss)
#         list_epoch.append(epoch)

#         # ====== Validation ====== #
#         model.eval() # Set the model be 'train mode' 
#         val_loss = 0 # to sum up each batch
        
#         with torch.no_grad():
#             for input_X, true_y in val_data:

#                 input_X = input_X.squeeze()
#                 input_X = input_X.view(-1, batch_size)

#                 pred_y = model(input_X)

#                 loss = criterion(pred_y.squeeze(), true_y)
#                 val_loss += loss.item()

#         val_loss = val_loss / len(val_data)
#         list_val_loss.append(val_loss)

#         # ====== Testing ====== #
#         model.eval() # Set the model be 'train mode' 
#         correct = 0 # to sum up each batch
        
#         with torch.no_grad():
#             for input_X, true_y in test_data:

#                 input_X = input_X.squeeze()
#                 input_X = input_X.view(-1, batch_size)

#                 pred_y = model(input_X).max(1, keepdim=True)[1].squeeze()
#                 correct += pred_y.eq(true_y).sum()

#             acc = correct.item() / len(test_data.dataset)
#             list_acc.append(acc)
#             list_acc_epoch.append(epoch)
        
#         print('Epoch: {}, Train Loss: {}, Val Loss: {}, Test Acc: {}%'.format(epoch, train_loss, val_loss, acc*100))

#     return list_epoch, list_train_loss, list_val_loss, list_acc, list_acc_epoch

# #------------------------------------------------------------------------------
# # Run the model 
# #------------------------------------------------------------------------------
# ts = time.time()

# seed = 123
# np.random.seed(seed)
# torch.manual_seed(seed)

# parser = argparse.ArgumentParser()

# args = parser.parse_args("")

# args.in_dim = window_size
# args.out_dim = 10

# args.hid_dim = 100
# args.n_layers = 5
# args.act = 'relu'
# args.dropout = 0.1
# args.use_bn = 'True'
# args.use_xavier = 'True'
# args.momentum = 0.9

# args.lr = 0.005
# args.epoch = 100

# list_epoch, list_train_loss, list_val_loss, list_acc, list_acc_epoch = trainModel(args)

# te = time.time()

# print('Elapsed time: {} sec'.format(int(te-ts)))



plt.figure(figsize=(12,4))
plt.xlim(x[-2*window_size], x[-1])
plt.grid(True)
plt.plot(x[-2*window_size:], y[-2*window_size:])
plt.plot(x[-window_size:], preds[window_size:], 'r-')
plt.show()