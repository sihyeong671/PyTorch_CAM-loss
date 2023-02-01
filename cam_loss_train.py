import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import *
from uilts import *


def train_and_test():
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    transform = transforms.Compose([
        # transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    mnist_train = datasets.MNIST(root='./mnist/',train=True,transform=transform,download=True)
    mnist_test = datasets.MNIST(root='./mnist/',train=False,transform=transform,download=True)

    BATCH_SIZE = 256
    
    train_iter = DataLoader(mnist_train,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
    test_iter = DataLoader(mnist_test,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)

    # torch.autograd.set_detect_anomaly(True)
    
    # ---TestModel---
    model = TestModel(10).to(device)
    optm_theta = optim.Adam(model.conv.parameters(),lr=1e-3)
    optm_w = optim.Adam(model.fc.parameters(), lr=1e-3)
    
    # ---Resnet18---
    # theta_params = [param for name, param in model.backbone.named_parameters() if 'fc' not in name]
    # optm_theta = optim.SGD(theta_params,lr=1e-2)
    # optm_w = optim.SGD(model.backbone.fc.parameters(), lr=1e-2)
    
    l_cam = nn.L1Loss()
    l_ce = nn.CrossEntropyLoss()
    
    model.train()
    EPOCHS = 50
    alpha = 0
    for epoch in range(1, EPOCHS+1):
        loss_sum = 0
        L_cam_sum = 0
        L_ce_sum = 0
        for batch_in, batch_out in tqdm(train_iter):
            inputs = batch_in.to(device)
            labels = batch_out.to(device)
            y_pred, feature_map = model.forward(inputs)

            if epoch >= 15:
                alpha = 3
            
            # CAAM, CAM
            CAAM = torch.sum(feature_map, dim=1) # B H W
            
            # ---TestModel---
            CAM = madeCAM(feature_map, labels, model.fc.weight.data.T) # B H W
            
            # ---Resnet18---
            # CAM = madeCAM(feature_map, labels, model.backbone.fc.weight.data.T) # B H W

            CAAM_normalize = minMaxNormalize(CAAM)
            CAM_noramlize = minMaxNormalize(CAM)
            
            L_cam_loss = l_cam(CAAM_normalize, CAM_noramlize) 
            L_ce_loss = l_ce(y_pred, labels)
            CAM_loss = alpha*L_cam_loss + L_ce_loss

            
            optm_theta.zero_grad()
            optm_w.zero_grad()
            CAM_loss.backward()
            optm_theta.step()
            optm_w.step()

            loss_sum += CAM_loss
            L_cam_sum += L_cam_loss
            L_ce_sum += L_ce_loss

        loss_avg = loss_sum/len(train_iter)
        cam_avg = L_cam_sum/len(train_iter)
        ce_avg = L_ce_sum/len(train_iter)

        if (epoch % 1) == 0:
            train_acc, f1_train = validation(model, train_iter, device)
            test_acc, f1_test = validation(model, test_iter, device)
            with open('cam_loss_log.txt', 'a') as f:
                f.write(f"epoch: {epoch} CAM_loss:{loss_avg:.3f}(L_cam: {cam_avg:.3f}, L_ce: {ce_avg:.3f}) train_acc: {train_acc:.3%} test_acc:{test_acc:.3%}, train_f1: {f1_train:.3f} test_f1: {f1_test:.3f}\n")

    # ---test---
    n_sample = 25
    sample_indices = np.random.choice(len(mnist_test.targets), n_sample, replace=False)
    test_x = mnist_test.data[sample_indices]
    test_y = mnist_test.targets[sample_indices]
    test_x.unsqueeze_(dim=1)
    with torch.no_grad():
        y_pred, _ = model(test_x.type(torch.float).to(device)/255.)
    y_pred = y_pred.argmax(axis=1)
    plt.figure(figsize=(10,10))
    for idx in range(n_sample):
        plt.subplot(5, 5, idx+1)
        plt.imshow(test_x[idx].squeeze(), cmap='gray')
        plt.axis('off')
        plt.title("Pred:%d, Label:%d"%(y_pred[idx],test_y[idx]))
    plt.savefig('cam_loss_result.png')    
    print("Done")

if __name__ == "__main__":
    seed_everythig(777)
    train_and_test()