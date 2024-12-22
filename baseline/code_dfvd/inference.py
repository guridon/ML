import os
import json
import pandas as pd
import torch

from tqdm import tqdm

def save_model_info(args):
    args_dict = vars(args)
    save_path = args.model_info_dir
    filename = f"{args.model}_{args.info}.json"
    full_path = os.path.join(save_path, filename)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with open(full_path, "w") as json_file:
        json.dump(args_dict, json_file, indent=4)

def inference(args, model, test_loader):
    model.to(args.device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in tqdm(iter(test_loader)):
            features = features.float().to(args.device)
            
            probs = model(features)

            probs  = probs.cpu().detach().numpy()
            predictions += probs.tolist()
    return predictions

def make_submission(args, preds):
    submit = pd.read_csv(args.sample_submission_file_path)
    submit.iloc[:, 1:] = preds
    
    save_path = args.csv_dir
    filename = f'submission_{args.info}.csv'
    full_path = os.path.join(args.csv_dir, filename)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    submit.to_csv(full_path, index=False)
    print(f"File saved to {save_path}")
    args.result_file_name = filename
    

    
    