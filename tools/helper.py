import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import torch
import torch.nn as nn
import time
import numpy as np
import torch.nn.functional as F
from .dtw_loss import SoftDTW


def simclr_loss(output_fast,output_slow,Temperature=0.1,normalize=True):
    out = torch.cat((output_fast, output_slow), dim=0)
    sim_mat = torch.mm(out, torch.transpose(out,0,1))
    if normalize:
        sim_mat_denom = torch.mm(torch.norm(out, dim=1).unsqueeze(1), torch.norm(out, dim=1).unsqueeze(1).t())
        sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)
    sim_mat = torch.exp(sim_mat / Temperature)
    if normalize:
        sim_mat_denom = torch.norm(output_fast, dim=1) * torch.norm(output_slow, dim=1)
        sim_match = torch.exp(torch.sum(output_fast * output_slow, dim=-1) / sim_mat_denom / Temperature)
    else:
        sim_match = torch.exp(torch.sum(output_fast * output_slow, dim=-1) / Temperature)
    sim_match = torch.cat((sim_match, sim_match), dim=0)
    norm_sum = torch.exp(torch.ones(out.size(0)) / Temperature )
    norm_sum = norm_sum.cuda()
    loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))
    return loss


def network_forward_train(base_model, taa, regressor, decoder, pred_scores, video_1, label_1, class_1, video_2, label_2, class_2, diff, mse, bce,
                          optimizer, opti_flag, epoch, batch_idx, batch_num, args):
    loss = 0.0
    start = time.time()
    optimizer.zero_grad()

    ############# I3D featrue #############
    feature_full, feature_thin = base_model(video_1, video_2)

    logits1, logits2, score1, score2, pred_class1, pred_class2 = taa(feature_full)
    logits3, logits4, score3, score4, pred_class3, pred_class4 = taa(feature_thin)
    logits_1 = torch.cat([logits1, logits3],dim=1)
    logits_2 = torch.cat([logits2, logits4], dim=1)

    decoder_12 = decoder(logits_1, logits_2)
    decoder_21 = decoder(logits_2, logits_1)
    decoder_1 = decoder(logits1, logits3)
    decoder_2 = decoder(logits2, logits4)
    decoder_all = torch.cat((decoder_12,decoder_21),dim=0)
    decoder_self = torch.cat((decoder_1,decoder_2),dim=0)

    total = torch.cat([logits1, logits2, logits3, logits4], dim=0)
    pred = regressor(decoder_all)
    pred_self = regressor(decoder_self)
    preds = regressor(total)

    pred = pred.mean(1)
    preds = preds.mean(1)
    pred_self = pred_self.mean(1)
    preds1, preds2, preds3, preds4 = torch.chunk(preds, 4, dim=0)
    pred1,pred2 = torch.chunk(pred,2,dim=0)
    pred_self1, pred_self2 = torch.chunk(pred_self, 2, dim=0)

    sdtw = SoftDTW(use_cuda=True, gamma=0.1)
    zero_lable = torch.zeros_like(pred_self1).cuda()
    loss1 = mse(logits1,logits3) + mse(logits2,logits4)
    loss2 = simclr_loss(logits1.mean(1),logits3.mean(1)) + simclr_loss(logits2.mean(1),logits4.mean(1))
    loss3 = mse(pred1,label_1 - label_2) + mse(pred2,label_2 - label_1)
    loss4 = mse(preds1, label_1) + mse(preds2, label_2) + mse(preds3, label_1) + mse(preds4, label_2)
    loss5 = mse(score1, label_1) + mse(score2, label_2) + mse(score3, label_1) + mse(score4, label_2)
    loss6 = bce(pred_class1, class_1) + bce(pred_class2, class_2) + bce(pred_class3, class_1) + bce(pred_class4, class_2)
    loss7 = 10. * (mse(pred_self1, zero_lable) + mse(pred_self2, zero_lable))
    loss8 = sdtw(logits1,logits2).mean() + sdtw(logits3, logits4).mean()
    loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8
    loss.backward()
    optimizer.step()

    end = time.time()
    batch_time = end - start
    # evaluate result of training phase
    if args.benchmark == 'MTL':
        if args.usingDD:
            score1 = (pred1 + label_2) * diff
            score2 = preds1 * diff
            score = (score1 + score2) / 2.
        else:
            score1 = pred1 + label_2
            score2 = preds1
            score = (score1 + score2) / 2.
    elif args.benchmark == 'Seven':
        score1 = pred1 + label_2
        score2 = preds1
        score = (score1 + score2) / 2.
    elif args.benchmark == 'FineDiving':
        score1 = pred1 + label_2
        score2 = preds1
        score = (score1 + score2) / 2.
    else:
        raise NotImplementedError()
    pred_scores.extend([i.item() for i in score])

    if batch_idx % args.print_freq == 0:
        print('[Training][%d/%d][%d/%d] \t Batch_time: %.2f \t Batch_loss: %.4f \t lr1 : %0.5f \t lr2 : %0.5f'
              % (epoch, args.max_epoch, batch_idx, batch_num, batch_time, loss.item(), optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))


def network_forward_test(base_model, taa, regressor, decoder, pred_scores,
                         video_1, video_2_list, label_2_list, diff, args):
    score = 0
    for video_2, label_2 in zip(video_2_list, label_2_list):

        ############# I3D featrue #############
        feature_full, feature_thin = base_model(video_1, video_2)
        logits1, logits2, score1, score2, pred_class1, pred_class2 = taa(feature_full)
        logits3, logits4, score3, score4, pred_class3, pred_class4 = taa(feature_thin)

        logits_1 = torch.cat([logits1, logits3], dim=1)
        logits_2 = torch.cat([logits2, logits4], dim=1)
        decoder_12 = decoder(logits_1, logits_2)
        decoder_21 = decoder(logits_2, logits_1)
        decoder_all = torch.cat((decoder_12, decoder_21), dim=0)

        total = torch.cat([logits1, logits2, logits3, logits4], dim=0)
        pred = regressor(decoder_all)
        preds = regressor(total)

        pred = pred.mean(1)
        preds = preds.mean(1)
        preds1, preds2, preds3, preds4 = torch.chunk(preds, 4, dim=0)
        pred1, pred2 = torch.chunk(pred, 2, dim=0)

        # evaluate result of training phase
        if args.benchmark == 'MTL':
            if args.usingDD:
                score1 = (pred1 + label_2) * diff
                score2 = preds1 * diff
                score += (score1 + score2) / 2.
            else:
                score1 = pred1 + label_2
                score2 = preds1
                score += (score1 + score2) / 2.
        elif args.benchmark == 'Seven':
            score1 = pred1 + label_2
            score2 = preds1
            score += (score1 + score2) / 2.
        elif args.benchmark == 'FineDiving':
            score1 = pred1 + label_2
            score2 = preds1
            score += (score1 + score2) / 2.
        else:
            raise NotImplementedError()
    pred_scores.extend([i.item() / len(video_2_list) for i in score])


def save_checkpoint(base_model, taa, regressor, decoder, optimizer, epoch,
                    epoch_best_aqa, rho_best, L2_min, RL2_min, prefix, args):
    torch.save({
        'base_model': base_model.state_dict(),
        'taa': taa.state_dict(),
        'regressor': regressor.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'epoch_best_aqa': epoch_best_aqa,
        'rho_best': rho_best,
        'L2_min': L2_min,
        'RL2_min': RL2_min,
    }, os.path.join(args.experiment_path, prefix + '.pth'))


def save_outputs(pred_scores, true_scores, args):
    save_path_pred = os.path.join(args.experiment_path, 'pred.npy')
    save_path_true = os.path.join(args.experiment_path, 'true.npy')
    np.save(save_path_pred, pred_scores)
    np.save(save_path_true, true_scores)
