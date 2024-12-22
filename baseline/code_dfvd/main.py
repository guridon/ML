from args import parse_args
from utils import seed_everything, get_mfcc_feature, get_model
from dataset import prepare_data, MFCCDataset
from dataloader import get_loaders
from utils import get_mfcc_feature, get_saved_feature
from model import MLP
from trainer import train
from inference import save_model_info, inference, make_submission

import torch
import wandb

def main(args):
    wandb.login()
    wandb.init(project='DACON', entity='fnelwndls')
    seed_everything(args.seed)
    # Load data 
    data = prepare_data(args=args)
    # Split data
    train_data, val_data = data.get_train_val()   # type: df
    
    # Extract_features
    if args.feature == False:
        train_mfcc, train_labels = get_mfcc_feature(args, train_data, train_mode=True)
        val_mfcc, val_labels = get_mfcc_feature(args, val_data, train_mode=True)
    else:
        train_mfcc, train_labels = get_saved_feature(args, train_data, train_mode=True)
        val_mfcc, val_labels = get_saved_feature(args, val_data, train_mode=True)
        
    # Dataset
    train_dataset = MFCCDataset(train_mfcc, train_labels)
    val_dataset = MFCCDataset(val_mfcc, val_labels)
    
    # Dataloader
    train_loader = get_loaders(args, train_dataset, train_mode = True)
    val_loader = get_loaders(args, val_dataset, train_mode = True)
    # load model & train
    model = get_model(args=args).to(args.device)
    optimizer = torch.optim.Adam(params = model.parameters(), lr = args.lr)
    infer_model = train(args, model, optimizer, train_loader, val_loader)
    
    # save model info
    save_model_info(args)
    
    ### inference ###
    test_data = data.get_test()
    if args.feature == False:
        test_mfcc = get_mfcc_feature(args, test_data, train_mode = False)
    else:
        test_mfcc = get_saved_feature(args, test_data, train_mode=False)
    test_dataset = MFCCDataset(test_mfcc, None)
    test_loader = get_loaders(args, test_dataset, train_mode = False)

    preds = inference(args, infer_model, test_loader)
    make_submission(args, preds)
    
    wandb.finish()
    
    
if __name__ == "__main__":
    args = parse_args()
    main(args)