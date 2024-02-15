import argparse

from module.utils import Config
from module.base_trainer import Trainer as BaseTrainer
from module.cam_loss_trainer import Trainer as CamLossTrainer


def run(config: Config):
    
    if config.use_cam:
        trainer = CamLossTrainer(config=config)
    else:
        trainer = BaseTrainer(config=config)
    
    trainer.setup()

    trainer.train()

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4) # if your os is Windows, then set 0
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./log")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model_name", type=str, default="custom")
    parser.add_argument("--use_cam", type=bool, default=False)

    args = parser.parse_args()
    
    config = Config(args)
    print(config)
    run(config)