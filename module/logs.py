import os
import logging
import torch
from torch.utils.tensorboard import SummaryWriter


def get_logger(name: str, **kwargs):
    if name == "tensorboard":
        # https://blog.naver.com/PostView.naver?blogId=wjddn9252&logNo=222371807209
        return SummaryWriter(**kwargs)
    elif name == "csv_logger":
        # csv logger
        return CSVLogger(**kwargs)
    else:
        ValueError(f"There is no {name} logging tool")


class CSVLogger:
    def __init__(
            self,
            log_dir: str,
            log_name: str="exp",
            log_level=logging.INFO,
            log_fmt: str="[%(asctime)s] %(levelname)s %(message)s"):
        
        os.makedirs(log_dir, exist_ok=True)
        formatter = logging.Formatter(log_fmt)
        handler = logging.FileHandler(os.path.join(log_dir, f"{log_name}.log"))
        handler.setFormatter(formatter)
        self.logger = logging.getLogger() # TODO : add logger_name
        self.logger.setLevel(log_level)
        self.logger.addHandler(handler)

    def write(self, msg: str):
        self.logger.info(msg)

    # TODO
    # change log level function


if __name__ == "__main__":
    pass

