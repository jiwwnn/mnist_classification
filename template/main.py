import dataset
from model import LeNet5, CustomMLP, LeNet5_Regularized
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

def train(model, trn_loader, device, criterion, optimizer):
    trn_loss_sum, trn_acc_sum, val_loss_sum, val_acc_sum = 0, 0, 0, 0
    trn_loss_lst, trn_acc_lst, val_loss_lst, val_acc_lst = [], [], [], []
    
    for epoch in range(60):
        model.train()
        
        for batch_idx, (data,target) in enumerate(trn_loader[0]):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # forward
            output = model(data)
            loss  = criterion(output, target)
            trn_loss_sum += loss.item()
            trn_loss_lst.append(loss.item())
            
            # backward
            loss.backward()
            optimizer.step()  
            
            # accuracy
            pred = torch.argmax(output, dim=1)
            acc = (pred == target).sum() / len(pred)
            trn_acc_sum += acc.item() 
            trn_acc_lst.append(acc.item())
    
        trn_loss = trn_loss_sum / (len(trn_loader[0]) * (epoch+1))
        acc = trn_acc_sum / (len(trn_loader[0]) * (epoch+1)) 
        
        # Validation
        model.eval()
        
        for batch_idx, (data,target) in enumerate(trn_loader[1]):
            data, target = data.to(device), target.to(device)

            # forward
            output = model(data)
            loss  = criterion(output, target)
            val_loss_sum += loss.item()
            val_loss_lst.append(loss.item())
            
            # accuracy
            pred = torch.argmax(output, dim=1)
            acc = (pred == target).sum() / len(pred)
            val_acc_sum += acc.item() 
            val_acc_lst.append(acc.item())
        
        val_loss = val_loss_sum / (len(trn_loader[1]) * (epoch+1))
        val_acc = val_acc_sum / (len(trn_loader[1]) * (epoch+1))
        
        print(f"Epoch {epoch:2d} | Train Loss : {trn_loss:.3f}, Train Accuracy : {acc:.3f}, Validataion Loss : {val_loss:.3f}, Validation Accuracy : {val_acc:.3f}")
    
    # Visualization
    plt.figure(figsize=(10,8))
    plt.subplot(2, 2, 1)
    plt.plot(trn_loss_lst)
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(2, 2, 2)
    plt.plot(trn_acc_lst)
    plt.title('Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(2, 2, 3)
    plt.plot(val_loss_lst)
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(2, 2, 4)
    plt.plot(val_acc_lst)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('LeNet5.png')

    return trn_loss, acc


def test(model, tst_loader, device, criterion):
    tst_loss_sum, tst_acc_sum = 0, 0
    model.eval()
    
    for batch_idx, (data,target) in enumerate(tst_loader):
        data, target = data.to(device), target.to(device)
        
        # forward
        output = model(data)
        loss  = criterion(output, target)
        tst_loss_sum += loss.item()
        
        # accuracy
        pred = torch.argmax(output, dim=1)
        acc = (pred == target).sum() / len(pred)
        tst_acc_sum += acc.item()

    tst_loss = tst_loss_sum / len(tst_loader) 
    acc = tst_acc_sum / len(tst_loader)
    print(f"Test Loss : {tst_loss:.3f}, Test Accuracy : {acc:.3f}")

    return tst_loss, acc


def main():
    batch_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainset = dataset.MNIST(data_dir='/dshome/ddualab/jiwon/mnist-classification/data/train/')
    testset = dataset.MNIST(data_dir = '/dshome/ddualab/jiwon/mnist-classification/data/test/')
    
    trn_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    tst_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    
    # LeNet5 # CustomMLP # LeNet5_Regularized
    model = LeNet5().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    trn_loss, trn_acc = train(model, (trn_loader, tst_loader), device, criterion, optimizer)
    tst_loss, tst_acc = test(model, tst_loader, device, criterion)

if __name__ == '__main__':
    main()
