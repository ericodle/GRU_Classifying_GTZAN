import json
import numpy as np
from sklearn.model_selection import train_test_split
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import functional as F

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

# Fetching the device that will be used throughout this notebook
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)



# path to json file that stores MFCCs and genre labels for each processed segment
DATA_PATH = "./MFCCs/"
FILENAME = "extrcted_data.json"


def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return  X, y

def train_val_test_split(X, y, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=True)
    return X_train, X_val, X_test, y_train, y_val, y_test
  
# load data
X, y = load_data(DATA_PATH+FILENAME)

# create train/test split
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, 0.1)

print("train, val, test mfccs")
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print("train, val, test targets(labels)")
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

from torch.utils.data import TensorDataset, DataLoader

tensor_X_train = torch.Tensor(X_train)
tensor_X_val = torch.Tensor(X_val)
tensor_y_train = torch.Tensor(y_train)
tensor_y_val = torch.Tensor(y_val)
tensor_X_test = torch.Tensor(X_test)
tensor_y_test = torch.Tensor(y_test)

train_dataset = TensorDataset(tensor_X_train,tensor_y_train)
val_dataset = TensorDataset(tensor_X_val,tensor_y_val)
test_dataset = TensorDataset(tensor_X_test,tensor_y_test)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

def accuracy(out, labels):

    _,pred = torch.max(out, dim=1)

    return torch.sum(pred==labels).item()
  
class CyclicLR(_LRScheduler):
    
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]
      
def cosine(t_max, eta_min=0):
    
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min)*(1 + np.cos(np.pi*t/t_max))/2
    
    return scheduler
  
# Define your model parameters here.
# Note: Parameters differ for MLP, CNN, and recurrent model types.
input_dim = 13    
hidden_dim = 256
layer_dim = 2
output_dim = 10
seq_dim = 130
dropout_prob = 0.2

train_acc = []
val_acc = []

# Training hyperparameters
lr = 0.001
n_epochs = 200
iterations_per_epoch = len(train_dataloader)
best_acc = 0
patience, trials = 20, 0

# Enter the model you wish to test here from corresponding class in ./models.py.
# Note 1: MLP, CNN, and recurrent models take different parameter arguments.
# Note 2: Different models may respond better to different optimizer functions.

model = ./models.GRU_model(input_dim, hidden_dim, layer_dim, output_dim, dropout_prob)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
opt = torch.optim.RMSprop(model.parameters(), lr=lr)
sched = CyclicLR(opt, cosine(t_max=iterations_per_epoch * 2, eta_min=lr/100))

print('Start model training')

for epoch in range(1, n_epochs + 1):

    
    tcorrect, ttotal = 0, 0
    for (x_batch, y_batch) in train_dataloader:
        model.train()
        x_batch, y_batch = [t.cuda() for t in (x_batch, y_batch)]
        y_batch = y_batch.to(torch.int64)
        sched.step()
        opt.zero_grad()
        out = model(x_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        opt.step()

        _,pred = torch.max(out, dim=1)
        ttotal += y_batch.size(0)
        tcorrect += torch.sum(pred==y_batch).item()

    train_acc.append(100 * tcorrect / ttotal)
    
    model.eval()
    vcorrect, vtotal = 0, 0
    for x_val, y_val in val_dataloader:
        x_val, y_val = [t.cuda() for t in (x_val, y_val)]
        out = model(x_val)
        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        vtotal += y_val.size(0)
        vcorrect += (preds == y_val).sum().item()
    
    vacc = vcorrect / vtotal
    val_acc.append(vacc*100)

    if epoch % 5 == 0:
        print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Val Acc.: {vacc:2.2%}')

    if vacc > best_acc:
        trials = 0
        best_acc = vacc
        torch.save(model, DATA_PATH+"model")
        print(f'Epoch {epoch} best model saved with val accuracy: {best_acc:2.2%}')
    else:
        trials += 1
        if trials >= patience:
            print(f'Early stopping on epoch {epoch}')
            break
            
    import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20,10))

plt.title("Train-Validation Accuracy")

plt.plot(train_acc, label='train')

plt.plot(val_acc, label='validation')

plt.xlabel('num_epochs', fontsize=12)

plt.ylabel('accuracy', fontsize=12)

plt.ylim((0,100))
# If necessary, you can expand the X-axis to the number of epochs by un-commenting the line below.
#plt.xlim((0,n_epochs))

plt.legend(loc='best')

# Script terminates after providing a diagnostic training graph.
