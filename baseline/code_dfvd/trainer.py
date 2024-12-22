import wandb
import numpy as np
from sklearn.metrics import roc_auc_score

from torch import nn
import torch

from tqdm import tqdm

def train(args, model, optimizer, train_loader, val_loader):
    model.to(args.device)
    criterion = nn.BCELoss().to(args.device)
    
    best_val_score = 0
    best_model = None
    
    for epoch in range(1, args.n_epochs+1):
        model.train()
        train_loss = []
        for features, labels in tqdm(iter(train_loader)):
            features = features.float().to(args.device)
            labels = labels.float().to(args.device)
            
            optimizer.zero_grad()
            
            output = model(features)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        _val_loss, _val_score = validation(args, model, criterion, val_loader)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val AUC : [{_val_score:.5f}]')
        
        wandb.log({
                    "Epoch": epoch,
                    "Train Loss": _train_loss,
                    "Validation Loss": _val_loss,
                    "Validation Accuracy": _val_score
                    })
        
        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model
    
    return best_model


def multiLabel_AUC(y_true, y_scores):
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        auc_scores.append(auc)
    mean_auc_score = np.mean(auc_scores)
    return mean_auc_score

def validation(args, model, criterion, val_loader):
    model.eval()
    val_loss, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for features, labels in tqdm(iter(val_loader)):
            features = features.float().to(args.device)
            labels = labels.float().to(args.device)
            
            probs = model(features)
            
            loss = criterion(probs, labels)

            val_loss.append(loss.item())

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        _val_loss = np.mean(val_loss)

        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        
        # Calculate AUC score
        auc_score = multiLabel_AUC(all_labels, all_probs)
    
    return _val_loss, auc_score

# test?