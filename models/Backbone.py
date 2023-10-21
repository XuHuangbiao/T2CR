import torch.nn as nn
import torch
from .i3d import I3D
import logging


class I3D_backbone(nn.Module):
    def __init__(self, I3D_class):
        super(I3D_backbone, self).__init__()
        print('Using I3D backbone')
        self.backbone = I3D(num_classes=I3D_class, modality='rgb', dropout_prob=0.5)

    def load_pretrain(self, I3D_ckpt_path):
        try:
            self.backbone.load_state_dict(torch.load(I3D_ckpt_path))
            print('loading ckpt done')
        except:
            logging.info('Ckpt path {} do not exists'.format(I3D_ckpt_path))
            pass

    def forward(self, video_1, video_2):

        total_video = torch.cat((video_1, video_2), 0)
        # start_idx = list(range(0, 90, 10))
        start_idx = [0, 10, 20, 30, 40, 50, 60, 70, 80, 86]
        start_idx_2 = [0, 12, 24, 36, 48, 60, 72, 84]
        video_pack = torch.cat([total_video[:, :, i: i + 16] for i in start_idx])
        video_pack_2 = torch.cat([total_video[:, :, i: i + 16] for i in start_idx_2])
        video_pack = torch.cat([video_pack, video_pack_2], dim=0)
        total_feature = self.backbone(video_pack)
        # Nt, C, T, H, W = total_feamap.size()
        total_feature = total_feature.reshape(len(start_idx) + len(start_idx_2), len(total_video), -1).transpose(0, 1)
        total_feature_full = total_feature[:, :10, :]
        total_feature_thin = total_feature[:, 10:, :]
        # total_feamap = total_feamap.reshape(len(start_idx), len(total_video), C, T, H, W).transpose(0, 1)
        # feature_full_1 = total_feature_full[:total_feature_full.shape[0] // 2]
        # feature_full_2 = total_feature_full[total_feature_full.shape[0] // 2:]
        # feature_thin_1 = total_feature_thin[:total_feature_thin.shape[0] // 2]
        # feature_thin_2 = total_feature_thin[total_feature_thin.shape[0] // 2:]
        # feamap_1 = total_feamap[:total_feamap.shape[0] // 2]
        # feamap_2 = total_feamap[total_feamap.shape[0] // 2:]
        # com_feature_12 = torch.cat((feature_1, feature_2), 1)
        # com_feamap_12 = torch.cat((feamap_1, feamap_2), 2)
        # return feature_full_1,feature_full_2,feature_thin_1,feature_thin_2
        return total_feature_full, total_feature_thin
