import librosa
import random
import numpy as np
import pandas as pd

import os
from tqdm import tqdm

import torch
from torch import nn

from model import MLP

# seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
# feature
def get_mfcc_feature(args, df, train_mode=True):
    features = []
    labels = []
    for _, row in tqdm(df.iterrows()):
        y, sr = librosa.load(row['path'], sr=args.sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=args.n_mfcc)
        mfcc = np.mean(mfcc.T, axis=0)
        features.append(mfcc)

        if train_mode:
            label = row['label']
            label_vector = np.zeros(args.n_classes, dtype=float)
            label_vector[0 if label == 'fake' else 1] = 1
            labels.append(label_vector)

    if train_mode:
        return features, labels
    return features


def get_saved_feature(args, df_selected, train_mode=True):
    features = df_selected['features'].tolist()
    labels = []
    if train_mode:
        for _, row in df_selected.iterrows():
            label = row['label']
            label_vector = np.zeros(args.n_classes, dtype=float)
            label_vector[0 if label == 'fake' else 1] = 1
            labels.append(label_vector)
        return features, labels
    return features

def extract_wav2vec2_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)  # Wav2Vec 2.0은 16kHz
    input_values = processor(y, return_tensors="pt", sampling_rate=sr).input_values
    with torch.no_grad():
        features = model(input_values).last_hidden_state
    return features.mean(dim=1).squeeze().numpy() 

def get_wav2vec2_feature(df, train_mode=True):
    features = []
    labels = []
    for _, row in tqdm(df.iterrows()):
        feature = extract_wav2vec2_features(row['path'])
        features.append(feature)

        if train_mode:
            label = row['label']
            label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
            label_vector[0 if label == 'fake' else 1] = 1
            labels.append(label_vector)
    if train_mode:
        return np.array(features), np.array(labels)
    return np.array(features)

def get_model(args) -> nn.Module:
    model_name = args.model.lower()
    model = {
            "mlp": MLP,
        }.get(model_name)(args)
    return model
