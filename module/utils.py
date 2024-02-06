import os
import random
import json
from datetime import datetime

import numpy as np
from sklearn.metrics import f1_score

import torch


class Config:
    """
    This Class is for env variables
    If mode is train, Config parameter is save in folder which name is self.log_dir

    Args
    args: parameter made by argparse library
    """
    def __init__(self, args):
        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # if you have more than 1 GPU, may be define variable for other GPU (actually using pytorch lightning is better)
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.lr = args.lr
        self.seed = args.seed
        self.data_path = args.data_path
        self.save_dir = args.save_dir
        if args.exp_name is None:
            now = datetime.now()
            self.exp_name = now.strftime("%Y-%m-%d,%H:%M:%S")
        else:
            self.exp_name = args.exp_name
        self.mode = args.mode
        self.model_name = args.model_name
        self.use_cam = args.use_cam
        
        self.log_dir = os.path.join(self.save_dir, self.exp_name)
        
        if self.mode == "train":
            self._save()
        
    def __str__(self):
        attr = vars(self)
        return "\n".join(f"{key}: {value}" for key, value in attr.items())
    
    def _save(self):
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            config = vars(self)
            json.dump(config, f, indent=4)
        

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def made_cam(feature_map, labels, weight):
    # B : label
    labels = labels.view(labels.size(0), -1)
    W = torch.stack([weight[:,labels[i]] for i in range(len(labels))]) # B C 1
    W = W.unsqueeze(dim=-1) # B C 1 1

    output = torch.mul(feature_map, W)
    output = torch.sum(output, dim=1)

    return output


def minmax_normalize(inputs):
    # B H W
    _min, _ = torch.min(inputs, dim=1)
    _max, _ = torch.max(inputs, dim=1)

    _max = torch.unsqueeze(_max, dim=-1)
    _min = torch.unsqueeze(_min, dim=-1)

    numerator = torch.sub(inputs, _min)
    denominator = torch.sub(_max, _min)
    output = torch.div(numerator, denominator)

    return output

    