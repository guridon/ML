import argparse
from datetime import datetime

def current_time():
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return current_time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str, help="cpu or gpu")
    parser.add_argument("--info", default=current_time(), type=str, help="record time")
    # --feature --> true
    parser.add_argument("--feature", action="store_true", help="using saved features.csv")
    
    # Audio args
    parser.add_argument("--sr", default=32000, type=int, help="Sample Rate")
    parser.add_argument("--n_mfcc", default=13, type=int, help="N_MFCC_coefficient")
    
    # path
    parser.add_argument("--root_folder", default='./', type=str, help="root")
    parser.add_argument("--data_path", default='./', type=str, help="data path")
    parser.add_argument("--feature_path", default='./', type=str, help="feature_csv file path")
    
    parser.add_argument("--sample_submission_file_path", default='./sample_submission.csv', type=str, help="sample submission file path")
    parser.add_argument("--csv_dir", default='./csv_dir', type=str, help="path to save csv file")
    # parser.add_argument("--model_dir", default='./modelzip', type=str, help="path to save model")
    parser.add_argument("--model_info_dir", default='./info_dir', type=str, help="path to save train_info")
    
    # training
    parser.add_argument("--n_classes", default=2, type=int, help="number of class")
    parser.add_argument("--batch_size", default=96, type=int, help="batch size")
    parser.add_argument("--n_epochs", default=5, type=int, help="number of epochs")
    parser.add_argument("--lr", default=3e-4, type=float, help="learning rate")
    parser.add_argument("--seed", default=42, type=int, help="seed")
    
    # model
    parser.add_argument("--model", default='MLP', type=str, help="model")
    parser.add_argument("--input_dim", default=13, type=int, help="input dim")
    parser.add_argument("--hidden_dim", default=128, type=int, help="hidden_dim")
    parser.add_argument("--output_dim", default=2, type=int, help="output_dim(n_classes)")   # 2 classes
    
    args = parser.parse_args()

    return args