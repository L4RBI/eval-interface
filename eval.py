import torch
from utils import load_checkpoint

import config
from  pytorch_msssim import MS_SSIM

import sys
#chooses what model to train
from resUnet import Generator


from time import localtime
import os
from PIL import Image
import config
from torchvision.utils import save_image
import os


def main():
    gen = Generator(init_weight=config.INIT_WEIGHTS).to(config.DEVICE)
    x = Image.open(sys.argv[1])
    x = config.transform(x).unsqueeze_(0).to(config.DEVICE)
    print(x.shape)
    print(sys.argv[2])
    load_checkpoint(sys.argv[2], gen)
    gen.eval()
    
    with torch.no_grad():
        y_fake = gen(x)
    save_image(x * 0.5 + 0.5, sys.argv[1])
    save_image(y_fake * 0.5 + 0.5, "predicted" + sys.argv[1])


if __name__ == "__main__":
    main()

    