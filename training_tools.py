
import os
import utils
import torch
import time
import einops
from einops import rearrange, repeat

import torch.nn.functional as F
from tqdm import tqdm

import evaluation_tools as ev_tools


@torch.no_grad()
def extract_features(feat_ext, videos, batch_sz=512):
    # Feature extraction process
    b, f = videos.shape[:2]
    videos = einops.rearrange(videos, 'b f h w c -> (b f) h w c')
    features = [feat_ext(batch.cuda()) for batch in utils.batching(videos, batch_sz)]
    features = torch.cat(features, 0)
    features = einops.rearrange(features, '(b f) r d -> b f r d', b=b)
    return features


@torch.no_grad()
def extract_DINO_features(DINO_feat_ext, videos, batch_sz=512):
    # Feature extraction process
    b, f = videos.shape[:2]
    videos = einops.rearrange(videos, 'b f h w c -> (b f) h w c')
    DINO_features = []
    for batch in utils.batching(videos, batch_sz):
        batch_on_cuda = batch.cuda()
        # input shape: [b, c, h, w]
        batch_on_cuda = einops.rearrange(batch_on_cuda, 'b h w c -> b c h w')
        DINO_features.append(DINO_feat_ext(batch_on_cuda))
    DINO_features = torch.cat(DINO_features, 0)
    DINO_features = einops.rearrange(DINO_features, '(b f) d -> b f d', b=b)
    return DINO_features


@torch.no_grad()
def extract_all_features(backbone_feat_ext, DINO_feat_ext, videos, batch_sz=512):
    features = extract_features(backbone_feat_ext, videos, batch_sz)
    DINO_features = extract_DINO_features(DINO_feat_ext, videos, batch_sz)
    return features, DINO_features


@torch.no_grad()
def get_pseudo_pos_neg_label_matrix(feature, top_rate=0.3, bottom_rate=0.3, refer_axis=-1):

    assert top_rate>0 and bottom_rate>0 and top_rate+bottom_rate<=1, 'top rate and bottom rate error'

    # calculate cosine similarity
    feature = F.normalize(feature, p=2, dim=-1)
    sim_matrix = torch.einsum('aik,bjk->aijb', feature, feature)
    sim_matrix = rearrange(sim_matrix, 'a i j b -> a b i j')
    size = sim_matrix.shape[refer_axis]

    # get top k instance
    top_k = max(round(size*top_rate), 1)
    topk_sim, topk_indices = torch.topk(sim_matrix, top_k, refer_axis)
    pseudo_pos_label_matrix = torch.zeros_like(sim_matrix, device=sim_matrix.device)
    # pseudo_pos_label_matrix is a mask, 1 for pos instance
    pseudo_pos_label_matrix = pseudo_pos_label_matrix.scatter(refer_axis, topk_indices, 1)

    # get bottom k instance
    bottom_k = max(round(size*bottom_rate), 1)
    bottomk_sim, bottomk_indices = torch.topk(sim_matrix, bottom_k, refer_axis, largest=False)
    pseudo_neg_label_matrix = torch.zeros_like(sim_matrix, device=sim_matrix.device)
    # pseudo_neg_label_matrix is a mask, 1 for neg instance
    pseudo_neg_label_matrix = pseudo_neg_label_matrix.scatter(refer_axis, bottomk_indices, 1)

    return pseudo_pos_label_matrix, pseudo_neg_label_matrix, topk_sim, bottomk_sim


def train_one_epoch_with_eval(loss_list, epoch, global_step, backbone_feat_extractor, DINO_feat_extractor, model, 
                    loader, optimizer, lr_scheduler, fp16_scaler, loss_criterion_dict, eval_dataset, eval_q_loader, eval_r_loader, writer, meters, args):

    if loss_list == ['QuadLinearAP', 'InfoNCE', 'SSHN']:
        global_step = train_one_epoch_qlap_nce_sshn_with_eval(
                epoch, global_step, backbone_feat_extractor, model, 
                loader, optimizer, lr_scheduler, fp16_scaler, loss_criterion_dict, eval_dataset, eval_q_loader, eval_r_loader, writer, meters, args)
    elif loss_list == ['TBInnerQuadLinearAP', 'QuadLinearAP', 'InfoNCE', 'SSHN']:
        global_step = train_one_epoch_innerqlap_qlap_nce_sshn_with_eval(
                epoch, global_step, backbone_feat_extractor, DINO_feat_extractor, model, 
                loader, optimizer, lr_scheduler, fp16_scaler, loss_criterion_dict, eval_dataset, eval_q_loader, eval_r_loader, writer, meters, args)
    else:
        raise Exception('Loss list {} is not allowed!'.format(loss_list))

    return global_step


def train_one_epoch_qlap_nce_sshn_with_eval(epoch, global_step, backbone_feat_extractor, model, 
                loader, optimizer, lr_scheduler, fp16_scaler, loss_criterion_dict, 
                eval_dataset, eval_q_loader, eval_r_loader, writer, meters, args):
    
    pbar = tqdm(loader, desc='epoch {}'.format(epoch), unit='iter') if args.gpu == 0 else loader
    # Loop for the epoch
    for idx, (videos, labels) in enumerate(pbar):
        optimizer.zero_grad()
        labels = labels.cuda()

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # Extract features for the video frames
            backbone_features = extract_features(backbone_feat_extractor, videos, args.batch_sz_fe)

            # Calculate similarities for each video pair in the batch
            frame_level_similarities, similarities, regularization_loss = model(backbone_features)

            # Calculate losses
            infonce_loss = loss_criterion_dict['InfoNCE'](similarities, labels)
            hardneg_loss, self_loss = loss_criterion_dict['SSHN'](similarities, labels)
            ap_loss = loss_criterion_dict['QuadLinearAP'](similarities, labels).mean()

            # Final loss
            loss = infonce_loss + 4*ap_loss + args.lambda_parameter * (hardneg_loss + self_loss) + args.r_parameter * regularization_loss

        # Update model weights
        lr_scheduler.step_update(global_step)
        if fp16_scaler is not None:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            loss.backward()
            optimizer.step()
        global_step += 1

        meters.update('total_loss', loss)
        meters.update('ql_ap_loss', ap_loss)
        meters.update('infonce_loss', infonce_loss)
        meters.update('sshn_loss', (hardneg_loss + self_loss))
        meters.update('reg_loss', regularization_loss)
     
        # Logging
        if args.gpu == 0:
            if global_step % 5 == 0:
                pbar.set_postfix(**meters.to_str())
            if global_step % args.log_step == 0 and len(meters) >= 10:
                if global_step % args.eval_step == 0:
                    model_name = 'model_iter_{}.pth'.format(global_step)
                    utils.save_model(args, model, optimizer, global_step, model_name)
                    model_path = os.path.join(args.experiment_path, model_name)
                    eval_results = ev_tools.eval_on_FIVR(model_path, eval_dataset, eval_q_loader, eval_r_loader)
                    utils.writer_log_with_eval(writer, model.module, meters, args.log_step, optimizer.param_groups[0]['lr'], eval_results,
                                    videos, backbone_features, global_step)
                else:
                    utils.writer_log(writer, model.module, meters, args.log_step, optimizer.param_groups[0]['lr'],
                                    videos, backbone_features, global_step)

    return global_step


def train_one_epoch_innerqlap_qlap_nce_sshn_with_eval(epoch, global_step, backbone_feat_extractor, DINO_feat_extractor, model, 
                loader, optimizer, lr_scheduler, fp16_scaler, loss_criterion_dict, 
                eval_dataset, eval_q_loader, eval_r_loader, writer, meters, args):
    
    pbar = tqdm(loader, desc='epoch {}'.format(epoch), unit='iter') if args.gpu == 0 else loader
    # Loop for the epoch
    for idx, (videos, labels) in enumerate(pbar):
        optimizer.zero_grad()
        labels = labels.cuda()

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # Extract features for the video frames
            backbone_features, DINO_features = extract_all_features(backbone_feat_extractor, DINO_feat_extractor, videos, args.batch_sz_fe)
            pseudo_pos_labels, pseudo_neg_labels, topk_sim, bottomk_sim = get_pseudo_pos_neg_label_matrix(DINO_features, args.pseudo_label_top_rate, args.pseudo_label_bottom_rate)
            
            # Calculate similarities for each video pair in the batch
            frame_level_similarities, similarities, regularization_loss = model(backbone_features)

            # Calculate losses
            inner_ap_loss = loss_criterion_dict['TBInnerQuadLinearAP'](frame_level_similarities, pseudo_pos_labels, pseudo_neg_labels, labels).mean()
            infonce_loss = loss_criterion_dict['InfoNCE'](similarities, labels)
            hardneg_loss, self_loss = loss_criterion_dict['SSHN'](similarities, labels)
            ap_loss = loss_criterion_dict['QuadLinearAP'](similarities, labels).mean()

            # Final loss
            loss = infonce_loss + args.inner_parameter*inner_ap_loss + 4*ap_loss + args.lambda_parameter * (hardneg_loss + self_loss) + args.r_parameter * regularization_loss

        # Update model weights
        lr_scheduler.step_update(global_step)
        if fp16_scaler is not None:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            loss.backward()
            optimizer.step()
        global_step += 1

        meters.update('total_loss', loss)
        meters.update('inner_ql_ap_loss', inner_ap_loss)
        meters.update('ql_ap_loss', ap_loss)
        meters.update('infonce_loss', infonce_loss)
        meters.update('sshn_loss', (hardneg_loss + self_loss))
        meters.update('reg_loss', regularization_loss)
     
        # Logging
        if args.gpu == 0:
            if global_step % 5 == 0:
                pbar.set_postfix(**meters.to_str())
            if global_step % args.log_step == 0 and len(meters) >= 10:
                if global_step % args.eval_step == 0:
                    model_name = 'model_iter_{}.pth'.format(global_step)
                    utils.save_model(args, model, optimizer, global_step, model_name)
                    model_path = os.path.join(args.experiment_path, model_name)
                    eval_results = ev_tools.eval_on_FIVR(model_path, eval_dataset, eval_q_loader, eval_r_loader)
                    utils.writer_log_with_eval(writer, model.module, meters, args.log_step, optimizer.param_groups[0]['lr'], eval_results,
                                    videos, backbone_features, global_step)
                else:
                    utils.writer_log(writer, model.module, meters, args.log_step, optimizer.param_groups[0]['lr'],
                                    videos, backbone_features, global_step)

    return global_step

