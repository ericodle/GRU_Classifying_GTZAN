import sys
sys.path.append('./')
#sys.path.append('./src/')

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset 
from torchvision import transforms

from src import models
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Fetching the device that will be used throughout this notebook
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

# path to json file that stores MFCCs and genre labels for each processed segment
DATA_PATH = "./MFCCs/"
FILENAME = "mfcc_data.json"

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

    return X, y

def test_ann_model(model, test_dataloader, device='cpu'):
    model.eval()    
    torch.no_grad()
    count = 0
    correct = 0
    true = []
    preds = []

    model = model.to(device)

    for X_testbatch, y_testbatch in test_dataloader:
        X_testbatch = X_testbatch.unsqueeze(1).to(device)
        y_testbatch = y_testbatch.to(device)

        y_val = model(X_testbatch)

        predicted = torch.max(y_val, 1)[1]

        count += y_testbatch.size(dim=0)
        correct += (predicted == y_testbatch).sum()

        true.append(y_testbatch)
        preds.append(predicted)

    ground_truth = torch.cat(true)
    predicted_genres = torch.cat(preds)
    accuracy = correct.item() / count

    return ground_truth, predicted_genres, accuracy

def test_recurrent_model(model, test_dataloader, device='cpu'):
    model.eval()
    count = 0
    correct = 0
    true = []
    preds = []

    model = model.to(device)

    with torch.no_grad():
        for X_testbatch, y_testbatch in test_dataloader:
            X_testbatch = X_testbatch.to(device)
            y_testbatch = y_testbatch.to(device)

            # Initialize hidden state and cell state on the same device as input data
            h0 = torch.zeros(model.layer_dim, X_testbatch.size(0), model.hidden_dim).to(device)
            c0 = torch.zeros(model.layer_dim, X_testbatch.size(0), model.hidden_dim).to(device)

            y_val = model(X_testbatch)

            predicted = torch.max(y_val, 1)[1]

            count += y_testbatch.size(dim=0)
            correct += (predicted == y_testbatch).sum()

            true.append(y_testbatch.cpu())
            preds.append(predicted.cpu().detach())

    ground_truth = torch.cat(true)
    predicted_genres = torch.cat(preds)
    accuracy = correct.item() / count

    return ground_truth, predicted_genres, accuracy


def test_svm_model(model, X_test, y_test):
    # Flatten the test data if it's three-dimensional
    if len(X_test.shape) == 3:
        num_samples, num_features, num_timesteps = X_test.shape
        X_test = X_test.reshape(num_samples, num_features * num_timesteps)

    # Make predictions on test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return y_test, y_pred, accuracy

def test_transformer_model(model, test_dataloader, device='cpu'):
    model.eval()
    count = 0
    correct = 0
    true = []
    preds = []

    for X_testbatch, y_testbatch in test_dataloader:
        X_testbatch = X_testbatch.to(device)
        y_testbatch = y_testbatch.to(device)

        X_testbatch = X_testbatch.permute(0, 1, 2)

        # Ensure model is also on the same device
        model = model.to(device)

        y_val = model(X_testbatch)

        predicted = torch.max(y_val, 1)[1]

        count += y_testbatch.size(0)
        correct += (predicted == y_testbatch).sum().item()

        true.append(y_testbatch.detach().cpu())
        preds.append(predicted.detach().cpu())

    ground_truth = torch.cat(true)
    predicted_genres = torch.cat(preds)
    accuracy = correct / count

    return ground_truth, predicted_genres, accuracy
def save_ann_confusion_matrix(ground_truth, predicted_genres, class_names, output_directory):
    arr = confusion_matrix(ground_truth.view(-1).detach().cpu().numpy(), predicted_genres.view(-1).detach().cpu().numpy())
    df_cm = pd.DataFrame(arr, class_names, class_names)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("Predictions")
    plt.ylabel("Ground Truths")
    plt.title('Confusion Matrix', fontsize=15)
    output_file = os.path.join(output_directory, 'confusion_matrix.png')
    plt.savefig(output_file)
    plt.close()


def save_svm_confusion_matrix(ground_truth, predicted_genres, class_names, output_directory):

    #ground_truth = ground_truth.flatten()
    #predicted_genres = predicted_genres.flatten()

    # Get unique labels from ground truth and predictions
    unique_labels = np.unique(np.concatenate((ground_truth, predicted_genres)))

    # Calculate confusion matrix
    arr = confusion_matrix(ground_truth, predicted_genres, labels=unique_labels)

    # Create DataFrame for visualization
    df_cm = pd.DataFrame(arr, index=class_names, columns=class_names)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn', cbar=False)
    plt.xlabel("Predictions")
    plt.ylabel("Ground Truths")
    plt.title('Confusion Matrix', fontsize=15)
    plt.xticks(rotation=45)  # Rotate x-axis labels if necessary
    plt.yticks(rotation=0)   # Rotate y-axis labels if necessary

    # Save confusion matrix to file
    output_file = os.path.join(output_directory, 'confusion_matrix.png')
    plt.savefig(output_file)
    plt.close()

def train_val_split(X, y, val_ratio):
    train_ratio = 1 - val_ratio
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio, shuffle=True)
    return X_train, X_val, y_train, y_val

def accuracy(out, labels):
    _, pred = torch.max(out, dim=1)
    return torch.sum(pred == labels).item()

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
        return eta_min + (base_lr - eta_min) * (1 + np.cos(np.pi * t / t_max)) / 2
    return scheduler

def train_svm(X_train, y_train, X_val, y_val, output_directory):
    # Flatten the input data if it's three-dimensional
    if len(X_train.shape) == 3:
        num_samples, num_features, num_timesteps = X_train.shape
        X_train = X_train.reshape(num_samples, num_features * num_timesteps)

    if len(X_val.shape) == 3:
        num_samples_val, _, _ = X_val.shape
        X_val = X_val.reshape(num_samples_val, num_features * num_timesteps)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train SVM
    svm_model = SVC()
    svm_model.fit(X_train_scaled, y_train)

    # Save the SVM model
    svm_model_path = os.path.join(output_directory, "svm_model.pkl")
    joblib.dump(svm_model, svm_model_path)
    print(f"SVM model saved at {svm_model_path}")

    # Get predictions on validation set
    y_pred = svm_model.predict(X_val_scaled)

    # Calculate classification report
    report = classification_report(y_val, y_pred)

    # Save classification report to a text file
    report_path = os.path.join(output_directory, "training_metrics.txt")
    with open(report_path, 'w') as file:
        file.write(report)
    print(f"Training metrics saved at {report_path}")

    return svm_model

def plot_learning_metrics(train_loss, val_loss, train_acc, val_acc, output_directory):
    epochs = range(1, len(train_loss) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=600)

    color = 'tab:red'
    orange = 'tab:orange' 
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, train_loss, label='Train Loss', color=color)
    ax1.plot(epochs, val_loss, label='Validation Loss', color=orange)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs, train_acc, label='Train Accuracy', color=color)
    ax2.plot(epochs, val_acc, label='Validation Accuracy', color='green')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout(rect=[0.05, 0.05, 0.9, 0.9])  # Adjusting layout to leave space for title and legend
    fig.legend(loc='upper left', bbox_to_anchor=(1,1))  # Moving legend outside the plot
    plt.title('Learning Metrics', pad=20)  # Adding padding to the title
    plt.savefig(os.path.join(output_directory, "learning_metrics.png"), bbox_inches='tight')  # Use bbox_inches='tight' to prevent cutting off
    plt.close()

def main(model_type, output_directory, initial_lr):
    # load data
    X, y = load_data(DATA_PATH + FILENAME)

    # Add diagnostic prints to check data dimensions
    print("Loaded data dimensions:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # create train/val split
    X_train, X_val, y_train, y_val = train_val_split(X, y, 0.2)

    tensor_X_train = torch.Tensor(X_train)
    tensor_X_val = torch.Tensor(X_val)
    tensor_y_train = torch.Tensor(y_train)
    tensor_y_val = torch.Tensor(y_val)

    tensor_X_test = torch.Tensor(X)
    tensor_y_test = torch.Tensor(y)

    train_dataset = TensorDataset(tensor_X_train, tensor_y_train)
    val_dataset = TensorDataset(tensor_X_val, tensor_y_val)

    test_dataset = TensorDataset(tensor_X_test, tensor_y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    train_loss = [] 
    val_loss = []   
    train_acc = []
    val_acc = []

    # Training hyperparameters
    lr = initial_lr
    n_epochs = 10000
    iterations_per_epoch = len(train_dataloader)
    best_acc = 0
    patience, trials = 20, 0

    # Initialize model based on model_type

    if model_type == "SVM":
        svm_model = train_svm(X_train, y_train, X_val, y_val, output_directory)
    elif model_type == 'MLP':
        model = models.MLP_model()
    elif model_type == 'CNN':
        model = models.CNN_model()
    elif model_type == 'LSTM':
        model = models.LSTM_model(input_dim=13, hidden_dim=256, layer_dim=2, output_dim=10, dropout_prob=0.2)
    elif model_type == 'GRU':
        model = models.GRU_model(input_dim=13, hidden_dim=256, layer_dim=2, output_dim=10, dropout_prob=0.2)
    elif model_type == "Transformer":
        model = models.Transformer_model(input_dim=13, hidden_dim=256, num_layers=4, num_heads=1, ff_dim=4, dropout=0.2, output_dim=10)
    else:
        raise ValueError("Invalid model_type. Supported types are: SVM, MLP, CNN, LSTM, and GRU.")

    if model_type != "SVM":
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.RMSprop(model.parameters(), lr=lr)
        sched = CyclicLR(opt, cosine(t_max=iterations_per_epoch * 2, eta_min=lr / 100))

        print(f'Training {model_type} model with learning rate of {initial_lr}.')

    if model_type == "MLP":
        for epoch in range(1, n_epochs + 1):
            tcorrect, ttotal = 0, 0
            running_train_loss = 0
            for (x_batch, y_batch) in train_dataloader:
                model.train()
                x_batch = x_batch.unsqueeze(1)
                x_batch, y_batch = [t.cuda() for t in (x_batch, y_batch)]
                y_batch = y_batch.to(torch.int64)        
                opt.zero_grad()
                out = model(x_batch)
                loss = criterion(out, y_batch)
                running_train_loss += loss.item()
                loss.backward()
                opt.step()
                sched.step()
                _,pred = torch.max(out, dim=1)
                ttotal += y_batch.size(0)
                tcorrect += torch.sum(pred==y_batch).item()
            train_acc.append(100 * tcorrect / ttotal)
            epoch_train_loss = running_train_loss / len(train_dataloader)
            train_loss.append(epoch_train_loss)
            model.eval()
            vcorrect, vtotal = 0, 0
            running_val_loss = 0
            for x_val, y_val in val_dataloader:
                x_val = x_val.unsqueeze(1)
                x_val, y_val = [t.cuda() for t in (x_val, y_val)]
                out = model(x_val)
                preds = F.log_softmax(out, dim=1).argmax(dim=1)
                vtotal += y_val.size(0)
                vcorrect += (preds == y_val).sum().item()
                running_val_loss += criterion(out, y_val.long()).item()
            vacc = vcorrect / vtotal
            val_acc.append(vacc*100)
            epoch_val_loss = running_val_loss / len(val_dataloader)
            val_loss.append(epoch_val_loss)
            if epoch % 5 == 0:
                print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Val Acc.: {vacc:2.2%}')
               
            if vacc > best_acc:
                trials = 0
                best_acc = vacc
                torch.save(model, os.path.join(output_directory, "model"))
                print(f'Epoch {epoch} best model saved with val accuracy: {best_acc:2.2%}')
            else:
                trials += 1
                if trials >= patience:
                    print(f'Early stopping on epoch {epoch}')
                    break

    if model_type == "CNN":
        for epoch in range(1, n_epochs + 1):
            tcorrect, ttotal = 0, 0
            running_train_loss = 0
            for (x_batch, y_batch) in train_dataloader:
                model.train()
                x_batch = x_batch.unsqueeze(1)
                x_batch, y_batch = [t.cuda() for t in (x_batch, y_batch)]
                y_batch = y_batch.to(torch.int64)
                opt.zero_grad()
                out = model(x_batch)
                loss = criterion(out, y_batch)
                running_train_loss += loss.item()
                loss.backward()
                opt.step()
                sched.step()
                _,pred = torch.max(out, dim=1)
                ttotal += y_batch.size(0)
                tcorrect += torch.sum(pred==y_batch).item()
            train_acc.append(100 * tcorrect / ttotal)
            epoch_train_loss = running_train_loss / len(train_dataloader)
            train_loss.append(epoch_train_loss)
            model.eval()
            vcorrect, vtotal = 0, 0
            running_val_loss = 0
            for x_val, y_val in val_dataloader:
                x_val = x_val.unsqueeze(1)
                x_val, y_val = [t.cuda() for t in (x_val, y_val)]
                out = model(x_val)
                preds = F.log_softmax(out, dim=1).argmax(dim=1)
                vtotal += y_val.size(0)
                vcorrect += (preds == y_val).sum().item()
                running_val_loss += criterion(out, y_val.long()).item()
            vacc = vcorrect / vtotal
            val_acc.append(vacc*100)
            epoch_val_loss = running_val_loss / len(val_dataloader)
            val_loss.append(epoch_val_loss)
            if epoch % 5 == 0:
                print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Val Acc.: {vacc:2.2%}')
            if vacc > best_acc:
                trials = 0
                best_acc = vacc
                torch.save(model, os.path.join(output_directory, "model"))
                print(f'Epoch {epoch} best model saved with val accuracy: {best_acc:2.2%}')
            else:
                trials += 1
                if trials >= patience:
                    print(f'Early stopping on epoch {epoch}')
                    break

    if model_type == "LSTM":
        for epoch in range(1, n_epochs + 1):
            tcorrect, ttotal = 0, 0
            running_train_loss = 0
            for (x_batch, y_batch) in train_dataloader:
                model.train()
                x_batch, y_batch = [t.cuda() for t in (x_batch, y_batch)]
                y_batch = y_batch.to(torch.int64)
                opt.zero_grad()
                out = model(x_batch)
                loss = criterion(out, y_batch)
                running_train_loss += loss.item()
                loss.backward()
                opt.step()
                sched.step()
                _,pred = torch.max(out, dim=1)
                ttotal += y_batch.size(0)
                tcorrect += torch.sum(pred==y_batch).item()
            train_acc.append(100 * tcorrect / ttotal)
            epoch_train_loss = running_train_loss / len(train_dataloader)
            train_loss.append(epoch_train_loss)
            model.eval()
            vcorrect, vtotal = 0, 0
            running_val_loss = 0
            for x_val, y_val in val_dataloader:
                x_val, y_val = [t.cuda() for t in (x_val, y_val)]
                out = model(x_val)
                preds = F.log_softmax(out, dim=1).argmax(dim=1)
                vtotal += y_val.size(0)
                vcorrect += (preds == y_val).sum().item()
                running_val_loss += criterion(out, y_val.long()).item()
            vacc = vcorrect / vtotal
            val_acc.append(vacc*100)
            epoch_val_loss = running_val_loss / len(val_dataloader)
            val_loss.append(epoch_val_loss)
            if epoch % 5 == 0:
                print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Val Acc.: {vacc:2.2%}')
            if vacc > best_acc:
                trials = 0
                best_acc = vacc
                torch.save(model, os.path.join(output_directory, "model"))
                print(f'Epoch {epoch} best model saved with val accuracy: {best_acc:2.2%}')
            else:
                trials += 1
                if trials >= patience:
                    print(f'Early stopping on epoch {epoch}')
                    break

    if model_type == "GRU":
        for epoch in range(1, n_epochs + 1):
            tcorrect, ttotal = 0, 0
            running_train_loss = 0
            for (x_batch, y_batch) in train_dataloader:
                model.train()
                x_batch, y_batch = [t.cuda() for t in (x_batch, y_batch)]
                y_batch = y_batch.to(torch.int64)
                opt.zero_grad()
                out = model(x_batch)
                loss = criterion(out, y_batch)
                running_train_loss += loss.item()
                loss.backward()
                opt.step()
                sched.step()
                _,pred = torch.max(out, dim=1)
                ttotal += y_batch.size(0)
                tcorrect += torch.sum(pred==y_batch).item()
            train_acc.append(100 * tcorrect / ttotal)
            epoch_train_loss = running_train_loss / len(train_dataloader)
            train_loss.append(epoch_train_loss)
            model.eval()
            vcorrect, vtotal = 0, 0
            running_val_loss = 0
            for x_val, y_val in val_dataloader:
                x_val, y_val = [t.cuda() for t in (x_val, y_val)]
                out = model(x_val)
                preds = F.log_softmax(out, dim=1).argmax(dim=1)
                vtotal += y_val.size(0)
                vcorrect += (preds == y_val).sum().item()
                running_val_loss += criterion(out, y_val.long()).item()
            vacc = vcorrect / vtotal
            val_acc.append(vacc*100)
            epoch_val_loss = running_val_loss / len(val_dataloader)
            val_loss.append(epoch_val_loss)
            if epoch % 5 == 0:
                print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Val Acc.: {vacc:2.2%}')
            if vacc > best_acc:
                trials = 0
                best_acc = vacc
                torch.save(model, os.path.join(output_directory, "model"))
                print(f'Epoch {epoch} best model saved with val accuracy: {best_acc:2.2%}')
            else:
                trials += 1
                if trials >= patience:
                    print(f'Early stopping on epoch {epoch}')
                    break

    if model_type == "Transformer":
        for epoch in range(1, n_epochs + 1):
            tcorrect, ttotal = 0, 0
            running_train_loss = 0
            for (x_batch, y_batch) in train_dataloader:
                model.train()
                x_batch, y_batch = [t.cuda() for t in (x_batch, y_batch)]
                y_batch = y_batch.to(torch.int64)
                opt.zero_grad()
                out = model(x_batch)
                loss = criterion(out, y_batch)
                running_train_loss += loss.item()
                loss.backward()
                opt.step()
                sched.step()
                _,pred = torch.max(out, dim=1)
                ttotal += y_batch.size(0)
                tcorrect += torch.sum(pred==y_batch).item()
            train_acc.append(100 * tcorrect / ttotal)
            epoch_train_loss = running_train_loss / len(train_dataloader)
            train_loss.append(epoch_train_loss)
            model.eval()
            vcorrect, vtotal = 0, 0
            running_val_loss = 0
            for x_val, y_val in val_dataloader:
                x_val, y_val = [t.cuda() for t in (x_val, y_val)]
                out = model(x_val)
                preds = F.log_softmax(out, dim=1).argmax(dim=1)
                vtotal += y_val.size(0)
                vcorrect += (preds == y_val).sum().item()
                running_val_loss += criterion(out, y_val.long()).item()
            vacc = vcorrect / vtotal
            val_acc.append(vacc*100)
            epoch_val_loss = running_val_loss / len(val_dataloader)
            val_loss.append(epoch_val_loss)
            if epoch % 5 == 0:
                print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Val Acc.: {vacc:2.2%}')
            if vacc > best_acc:
                trials = 0
                best_acc = vacc
                torch.save(model, os.path.join(output_directory, "model"))
                print(f'Epoch {epoch} best model saved with val accuracy: {best_acc:2.2%}')
            else:
                trials += 1
                if trials >= patience:
                    print(f'Early stopping on epoch {epoch}')
                    break

    print("Training finished!")

    #Evaluate trained model

    if model_type == "SVM":
        # Test the model
        ground_truth, predicted_genres, accuracy = test_svm_model(svm_model, tensor_X_test, tensor_y_test)

        # Print test accuracy
        print(f'Test accuracy: {accuracy * 100:.2f}%')

        # Plot confusion matrix
        class_names = ['pop', 'classical', 'jazz', 'hiphop', 'reggae', 'disco', 'metal', 'country', 'blues', 'rock']
        save_svm_confusion_matrix(ground_truth, predicted_genres, class_names, output_directory)

    if model_type == "MLP":
        plot_learning_metrics(train_loss, val_loss, train_acc, val_acc, output_directory)
        print("Learning metrics plotted!")

        # Test the model
        ground_truth, predicted_genres, accuracy = test_ann_model(model, test_dataloader)

        # Print test accuracy
        print(f'Test accuracy: {accuracy * 100:.2f}%')

        # Plot confusion matrix
        class_names = ['pop', 'classical', 'jazz', 'hiphop', 'reggae', 'disco', 'metal', 'country', 'blues', 'rock']
        save_ann_confusion_matrix(ground_truth, predicted_genres, class_names, output_directory)

    if model_type == "CNN":
        plot_learning_metrics(train_loss, val_loss, train_acc, val_acc, output_directory)
        print("Learning metrics plotted!")

        # Test the model
        ground_truth, predicted_genres, accuracy = test_ann_model(model, test_dataloader)

        # Print test accuracy
        print(f'Test accuracy: {accuracy * 100:.2f}%')

        # Plot confusion matrix
        class_names = ['pop', 'classical', 'jazz', 'hiphop', 'reggae', 'disco', 'metal', 'country', 'blues', 'rock']
        save_ann_confusion_matrix(ground_truth, predicted_genres, class_names, output_directory)

    if model_type == "LSTM":
        plot_learning_metrics(train_loss, val_loss, train_acc, val_acc, output_directory)
        print("Learning metrics plotted!")

        # Test the model
        ground_truth, predicted_genres, accuracy = test_recurrent_model(model, test_dataloader, device=device)

        # Print test accuracy
        print(f'Test accuracy: {accuracy * 100:.2f}%')

        # Plot confusion matrix
        class_names = ['pop', 'classical', 'jazz', 'hiphop', 'reggae', 'disco', 'metal', 'country', 'blues', 'rock']
        save_ann_confusion_matrix(ground_truth, predicted_genres, class_names, output_directory)

    if model_type == "GRU":
        plot_learning_metrics(train_loss, val_loss, train_acc, val_acc, output_directory)
        print("Learning metrics plotted!")

        # Test the model
        ground_truth, predicted_genres, accuracy = test_recurrent_model(model, test_dataloader, device=device)

        # Print test accuracy
        print(f'Test accuracy: {accuracy * 100:.2f}%')

        # Plot confusion matrix
        class_names = ['pop', 'classical', 'jazz', 'hiphop', 'reggae', 'disco', 'metal', 'country', 'blues', 'rock']
        save_ann_confusion_matrix(ground_truth, predicted_genres, class_names, output_directory)

    if model_type == "Transformer":
        plot_learning_metrics(train_loss, val_loss, train_acc, val_acc, output_directory)
        print("Learning metrics plotted!")

        # Test the model
        ground_truth, predicted_genres, accuracy = test_transformer_model(model, test_dataloader)

        # Print test accuracy
        print(f'Test accuracy: {accuracy * 100:.2f}%')

        # Plot confusion matrix
        class_names = ['pop', 'classical', 'jazz', 'hiphop', 'reggae', 'disco', 'metal', 'country', 'blues', 'rock']
        save_ann_confusion_matrix(ground_truth, predicted_genres, class_names, output_directory)


if __name__ == '__main__':
    # Pass arguments via command line arguments when running the script
    model_type = sys.argv[1]  # e.g., "MLP", "CNN", "LSTM", "GRU", or "Transformer"
    output_directory = sys.argv[2]  # e.g., "./output/"
    initial_lr = float(sys.argv[3])  # e.g., 0.001

    # Call main function with provided arguments
    main(model_type, output_directory, initial_lr)

