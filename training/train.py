import os, json, pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from tokenizer import build_vocab
from dataset import HateDataset
from model import TextCNN

DATA_DIR = "data/processed"
ART_DIR = "backend/artifacts"
os.makedirs(ART_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
