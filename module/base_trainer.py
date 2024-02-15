import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from module.datasets import get_dataset
from module.models import get_model
from module.logs import get_logger
from module.utils import seed_everything, Config


class Trainer:
    def __init__(self, config: Config):
        self.config = config
    

    def setup(self, mode="train"):
        """
        you need to code how to get data
        and define dataset, dataloader, transform in this function
        """

        seed_everything(self.config.seed)
        
        # Model
        self.model = get_model(
            name=f"{self.config.model_name}",
            num_classes=10
        )
        self.model.to(self.config.device)

        if mode == "train":

            self.logger = get_logger(
                name="csv_logger",
                log_dir=f"{self.config.log_dir}",    
            )

            ## TODO ##
            # Hint : get data by using pandas or glob
            
            # Train
            train_transform = transforms.Compose([
                # add augmentation
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
            ])

            train_dataset = get_dataset(
                "mnist",
                root=f"{self.config.data_path}/minst",
                train=True,
                transform=train_transform,
                download=True
            )

            self.train_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                shuffle=True,
            )
            
            # Validation
            val_transform = transforms.Compose([
                transforms.ToTensor(), # 0 ~ 1
                transforms.Normalize((0.5), (0.5)) # -1 ~ 1
            ])

            val_dataset = get_dataset(
                "mnist",
                root=f"{self.config.data_path}/minst",
                train=False,
                transform=val_transform,
                download=True
            )         

            self.val_dataloader = DataLoader(
                dataset=val_dataset,
                batch_size=self.config.batch_size * 2,
                num_workers=self.config.num_workers,
                shuffle=False,
            )

            # Loss function
            self.loss_fn = nn.CrossEntropyLoss()

            # Optimizer
            self.optimizer = optim.Adam(self.model.parameters(),lr=self.config.lr)

            # LR Scheduler
            self.lr_scheduler = None

        elif mode == "test":

            test_transform = transforms.Compose([
                transforms.ToTensor(), # 0 ~ 1
                transforms.Normalize((0.5), (0.5)) # -1 ~ 1
            ])

            self.test_dataset = get_dataset(
                "mnist",
                root=f"{self.config.data_path}/minst",
                train=False,
                transform=test_transform,
                download=True
            )

            ckpt = torch.load(f"{self.config.log_dir}/{self.config.model_name}.pth", map_location=self.config.device)
            self.model.load_state_dict(ckpt)
            
            # self.test_dataloader = DataLoader(
            #     dataset=test_dataset,
            #     batch_size=self.config.batch_size * 2,
            #     num_workers=self.config.num_workers,
            #     shuffle=False,
            # )
    

    def train(self):
        
        
        # early stopping
        early_stopping = 0

        # metric
        # best_acc = 0
        best_f1 = 0

        # best_model = None
        
        for epoch in range(1, self.config.epochs+1):
            self.model.train()
            train_loss = 0
            train_acc = 0
            train_f1 = 0
            for imgs, labels in tqdm(self.train_dataloader):
                
                ## TODO ##
                # ----- Modify Example Code -----
                # following code is pesudo code
                # modify the code to fit your task 
                imgs = imgs.to(self.config.device)
                labels = labels.to(self.config.device)
                
                self.optimizer.zero_grad()

                pred, _ = self.model(imgs)
                loss = self.loss_fn(pred, labels)
                loss.backward()
                
                with torch.no_grad():
                    train_loss += loss.item()
                    _, y_pred = torch.max(pred.data, dim=1)
                    y_pred = y_pred.detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()
                    train_acc += accuracy_score(y_pred, labels)
                    train_f1 += f1_score(y_pred, labels, average="macro")
                
                self.optimizer.step()
                # -------------------------------
            train_acc /= len(self.train_dataloader)
            train_loss /= len(self.train_dataloader)
            train_f1 /= len(self.train_dataloader)

            val_loss, val_acc, val_f1 = self._valid()

            self.logger.write(f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
            self.logger.write(f"Epoch: {epoch}, Val loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.model.state_dict(), f"{self.config.log_dir}/{self.config.model_name}.pth")
            
            if early_stopping >= 5:
                break
            
            
    def _valid(self):
        # metric

        self.model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            val_f1 = 0
            for imgs, labels in tqdm(self.val_dataloader):
                imgs = imgs.to(self.config.device)
                labels = labels.to(self.config.device)

                pred, _ = self.model(imgs)
                loss = self.loss_fn(pred, labels)
                val_loss += loss.item()
                _, y_pred = torch.max(pred.data, dim=1)
                y_pred = y_pred.cpu().numpy()
                labels = labels.cpu().numpy()
                val_acc += accuracy_score(y_pred, labels)
                val_f1 += f1_score(y_pred, labels, average="macro")
            val_loss /= len(self.val_dataloader)
            val_acc /= len(self.val_dataloader)
            val_f1 /= len(self.val_dataloader)
        return val_loss, val_acc, val_f1
        
        
    def test(self):
        n_sample = 25
        sample_indices = np.random.choice(len(self.test_dataset.targets), n_sample, replace=False)
        test_x = self.test_dataset.data[sample_indices]
        test_y = self.test_dataset.targets[sample_indices]
        test_x.unsqueeze_(dim=1)
        with torch.no_grad():
            y_pred, _ = self.model(test_x.type(torch.float).to(self.config.device))
        y_pred = y_pred.argmax(axis=1)
        plt.figure(figsize=(10,10))
        for idx in range(n_sample):
            plt.subplot(5, 5, idx+1)
            plt.imshow(test_x[idx].squeeze(), cmap='gray')
            plt.axis('off')
            plt.title("Pred:%d, Label:%d"%(y_pred[idx],test_y[idx]))
        
        plt.savefig('result_without_CAMloss.png')
        print("Done")
    
    