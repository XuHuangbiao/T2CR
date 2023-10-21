import os
import sys
import numpy as np
import torch
import torch.nn as nn
import random
from tools import train_net, test_net
from utils import parser


def main():
    args = parser.get_args()
    parser.setup(args)
    seed = args.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    if args.benchmark == 'MTL':
        if not args.usingDD:
            args.score_range = 100
    print(args)
    if args.test:
        test_net(args)
    else:
        train_net(args)


if __name__ == '__main__':
    main()
