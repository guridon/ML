from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import os

# data load && split
class prepare_data():
    def __init__(self, args, train_mode = True):
        self.seed = args.seed
        self.data_path = args.data_path
        self.feature_path = args.feature_path
        self.df = self.using_saved_feat(args) if args.feature else self.load_data()
        self.test_df = self.using_saved_feat(args, train_mode = False) if args.feature else self.load_data(train_mode = False)
        self.train_data, self.val_data = self.split_train_val(self.df)
        
    def load_data(self, train_mode = True):
        if train_mode:
            df = pd.read_csv(os.path.join(self.data_path,"train.csv"))
            return df
        df = pd.read_csv(os.path.join(self.data_path,"test.csv"))
        return df
    
    def split_train_val(self, df):
        train, val, _, _= train_test_split(df, df['label'], test_size=0.2, random_state=self.seed)
        return train, val
    
    def using_saved_feat(self, args, train_mode = True):
        if train_mode:
            df = pd.read_csv(os.path.join(self.feature_path,"train_features.csv"))
            selected_columns = [
                                'id', 'label', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5',
                                'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 
                                'mfcc_12', 'mfcc_13'
                                ]
            args.selected_columns = selected_columns
            i=2
        else:
            df = pd.read_csv(os.path.join(self.feature_path,"test_features.csv"))
            selected_columns = [
                                'id', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5',
                                'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 
                                'mfcc_12', 'mfcc_13'
                                ]
            i=1

        df_selected = df[selected_columns]
        def convert_to_numpy(row):
            mfcc_features = row[i:].values.astype(np.float32) 
            return np.array(mfcc_features)
        
        df_selected = df_selected.copy()
        df_selected['features'] = df_selected.apply(convert_to_numpy, axis=1)
        
        drop_columns = [
        'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5',
        'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 
        'mfcc_12', 'mfcc_13'
            ]
        df_selected = df_selected.drop(drop_columns, axis=1)

        return df_selected
    
    def get_train_val(self):
        return self.train_data, self.val_data

    def get_test(self):
        return self.test_df
    
     
class MFCCDataset(Dataset):
    def __init__(self, mfcc, label):
        self.mfcc = mfcc
        self.label = label

    def __len__(self):
        return len(self.mfcc)

    def __getitem__(self, index):
        if self.label is not None:
            return self.mfcc[index], self.label[index]
        return self.mfcc[index]
    
    