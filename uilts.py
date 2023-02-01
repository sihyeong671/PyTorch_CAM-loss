import numpy as np
import torch
import os
import random
from sklearn.metrics import f1_score

def seed_everythig(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def validation(model, data_iter, device):
    with torch.no_grad():
        model.eval()
        n_total, n_correct, f1 = 0, 0, 0
        for batch_in,batch_out in data_iter:
            x_trgt = batch_in.to(device)
            y_trgt = batch_out.to(device)
            model_pred, _ = model(x_trgt)
            _, y_pred = torch.max(model_pred.data, 1)
            n_correct += (y_pred==y_trgt).sum().item()
            n_total += x_trgt.size(0)
            f1 += f1_score(y_trgt.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), average='macro')
        val_acc = (n_correct/n_total)
        f1 = f1/len(data_iter)
        model.train()
    return val_acc, f1


def madeCAM(feature_map, labels, weight):
    # B : label
    labels = labels.view(labels.size(0), -1)
    W = torch.stack([weight[:,labels[i]] for i in range(len(labels))]) # B C 1
    W = W.unsqueeze(dim=-1) # B C 1 1

    output = torch.mul(feature_map, W)
    output = torch.sum(output, dim=1)

    return output


def minMaxNormalize(inputs):
    # B H W
    _min, _ = torch.min(inputs, dim=1)
    _max, _ = torch.max(inputs, dim=1)

    _max = torch.unsqueeze(_max, dim=-1)
    _min = torch.unsqueeze(_min, dim=-1)

    numerator = torch.sub(inputs, _min)
    denominator = torch.sub(_max, _min)
    output = torch.div(numerator, denominator)

    return output

    