import os
import sys
# sys.path.append(os.getcwd())
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import torch
import torch.optim as optim
from models import I3D_backbone
from models import TAA
from models import MLP_score
from models import decoder_fuser
from utils.misc import import_class
from torch_videovision.torchvideotransforms import video_transforms, volume_transforms


def get_video_trans():
    train_trans = video_transforms.Compose([
        video_transforms.RandomHorizontalFlip(),
        video_transforms.Resize((228, 128)),
        video_transforms.RandomCrop(112),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_trans = video_transforms.Compose([
        video_transforms.Resize((228, 128)),
        video_transforms.CenterCrop(112),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_trans, test_trans


def dataset_builder(args):
    train_trans, test_trans = get_video_trans()
    Dataset = import_class("datasets." + args.benchmark)
    train_dataset = Dataset(args, transform=train_trans, subset='train')
    test_dataset = Dataset(args, transform=test_trans, subset='test')
    return train_dataset, test_dataset


def model_builder(args):
    base_model = I3D_backbone(I3D_class=400)
    base_model.load_pretrain(args.pretrained_i3d_weight)
    taa = TAA()
    regressor = MLP_score()
    Decoder_vit = decoder_fuser(dim=64, num_heads=8, num_layers=3)
    return base_model, taa, regressor, Decoder_vit


def build_opti_sche(base_model, taa, regressor, decoder, args):
    if args.optimizer == 'Adam':
        optimizer = optim.Adam([
            {'params': base_model.parameters(), 'lr': args.base_lr * args.lr_factor},
            {'params': taa.parameters()},
            {'params': regressor.parameters()},
            {'params': decoder.parameters()}
        ], lr=args.base_lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError()

    scheduler = None
    return optimizer, scheduler


def resume_train(base_model, taa, regressor, decoder, optimizer, args):
    ckpt_path = os.path.join(args.experiment_path, 'last.pth')
    if not os.path.exists(ckpt_path):
        print('no checkpoint file from path %s...' % ckpt_path)
        return 0, 0, 0, 1000, 1000
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    # parameter resume of base model
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)

    taa_ckpt = {k.replace("module.", ""): v for k, v in state_dict['taa'].items()}
    taa.load_state_dict(taa_ckpt)

    regressor_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor'].items()}
    regressor.load_state_dict(regressor_ckpt)

    decoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict['decoder'].items()}
    decoder.load_state_dict(decoder_ckpt)

    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

    # parameter
    start_epoch = state_dict['epoch'] + 1
    epoch_best_aqa = state_dict['epoch_best_aqa']
    rho_best = state_dict['rho_best']
    L2_min = state_dict['L2_min']
    RL2_min = state_dict['RL2_min']

    return start_epoch, epoch_best_aqa, rho_best, L2_min, RL2_min


def load_model(base_model, taa, regressor, decoder, args):
    ckpt_path = args.ckpts
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    # parameter resume of base model
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)

    taa_ckpt = {k.replace("module.", ""): v for k, v in state_dict['taa'].items()}
    taa.load_state_dict(taa_ckpt)

    regressor_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor'].items()}
    regressor.load_state_dict(regressor_ckpt)

    decoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict['decoder'].items()}
    decoder.load_state_dict(decoder_ckpt)

    epoch_best_aqa = state_dict['epoch_best_aqa']
    rho_best = state_dict['rho_best']
    L2_min = state_dict['L2_min']
    RL2_min = state_dict['RL2_min']
    print('ckpts @ %d epoch(rho = %.4f, L2 = %.4f , RL2 = %.4f)' % (epoch_best_aqa - 1, rho_best, L2_min, RL2_min))
    return
