import utils
import argparse
import datetime
import torch
import time
import torch.nn as nn
import numpy as np

import training_tools as tr_tools
import evaluation_tools as ev_tools

from tqdm import tqdm
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from datasets import augmentations
from datasets.generators import SSLGenerator
from model.losses import loss_select, InfoNCELoss, SSHNLoss
from model.similarity_network import SimilarityNetwork
from model.feature_extractor import FeatureExtractor
from model.dino_feature_extractor import DINOFeatureExtractor

def main(args):
    # Initialization of distributed processing
    utils.init_distributed_mode(args)

    print('\n> Create augmentations: {}'.format(args.augmentations))
    # Instantiation of the objects for weak and strong augmentations
    weak_aug = augmentations.WeakAugmentations(**vars(args))
    strong_aug = augmentations.StrongAugmentations(**vars(args))
    print(*[weak_aug, strong_aug], sep='\n')

    # Initialization of data generator and data loader
    print('\n> Create generator')
    dataset = SSLGenerator(weak_aug=weak_aug, strong_aug=strong_aug, **vars(args))
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, num_workers=args.workers, batch_size=args.batch_sz,
                        collate_fn=dataset.collate_fn, sampler=sampler, drop_last=True)
    args.epochs = int(args.iter_epochs // len(loader) + 1)

    # Initialization of the feature extraction network and similarity network
    print('\n> Building network')
    backbone_feat_extractor = FeatureExtractor[args.backbone.upper()].get_model(args.dims).cuda().eval()
    DINO_feat_extractor = None
    if args.use_pseudo_label:
        DINO_feat_extractor = DINOFeatureExtractor['DINO'].get_model(args.pretrained_DINO_path).cuda().eval()

    model = SimilarityNetwork[args.similarity_network].get_model(**vars(args))
    model = nn.parallel.DistributedDataParallel(model.cuda()) # only similarity network is trainable
    print(model)

    # Instantiation of losses
    loss_criterion_dict = {}
    for loss_name in args.loss_list:
        assert loss_name in args.loss_allowed, 'Input loss {} is not allowed.'.format(loss_name)
        loss_criterion_dict[loss_name] = loss_select(loss_name, args)
    print('loss_list: ', args.loss_list)

    # Initialization of the optimizer and lr scheduler
    params = [v for v in filter(lambda p: p.requires_grad, model.parameters())]

    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.iter_epochs,
        lr_min=args.final_lr*args.learning_rate,
        warmup_t=args.warmup_iters,
        warmup_lr_init=args.warmup_lr_init*args.learning_rate,
        t_in_epochs=False,
    )
    global_step = 0

    # Initialization of the FP16 Scaler if used
    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    # Load a saved model
    if args.load_model:
        global_step = utils.load_model(args, model, optimizer, args.saved_model_path)

    # Initialization of the reporting tools
    meters = utils.AverageMeterDict()
    writer = SummaryWriter(args.experiment_path) if args.gpu == 0 else None

    eval_dataset, eval_q_loader, eval_r_loader = ev_tools.get_eval_data()


    print('\n> Start training for {} epochs'.format(args.epochs))
    # Training loop
    for epoch in range(global_step // len(loader), args.epochs):
        sampler.set_epoch(epoch)
        dataset.next_epoch()
        meters.reset()
        model.train()

        global_step = tr_tools.train_one_epoch_with_eval(args.loss_list, epoch, global_step, backbone_feat_extractor, DINO_feat_extractor, model, 
                    loader, optimizer, lr_scheduler, fp16_scaler, loss_criterion_dict, eval_dataset, eval_q_loader, eval_r_loader, writer, meters, args)
    
        model.eval()

        # save model at the end of each epoch
        if args.gpu == 0:
            utils.save_model(args, model, optimizer, global_step, 'model.pth')



if __name__ == '__main__':
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(
        description='This is the code for the training of HAP-VR.',
        formatter_class=formatter)
    # Experiment arguments
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to frame files of the trainset.')
    parser.add_argument('--experiment_path', type=str, required=True,
                        help='Path of the experiment where the weights of the trained network and all logs will be stored.')
    parser.add_argument('--workers', default=12, type=int,
                        help='Number of workers used for the training.')
    parser.add_argument('--load_model', type=utils.bool_flag, default=False,
                        help='Boolean flag indicating that the weights from an existing model will be loaded.')
    parser.add_argument('--saved_model_path', type=str, help='The path of saved model path')
    
    parser.add_argument('--log_step', type=int, default=100,
                        help='Number of steps to save logs.')
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False,
                        help='Boolean flag indicating that fp16 scaling will be used.')
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training.')

    # Similarity network options
    parser.add_argument('--backbone', type=str, default='resnet', choices=[x.name.lower() for x in FeatureExtractor],
                        help='Backbone network used for feature extraction.')
    parser.add_argument('--similarity_network', type=str, default='ViSiL', choices=[x.name for x in SimilarityNetwork],
                        help='Similarity network used for similarity calculation.')
    parser.add_argument('--dims', type=int, default=512,
                        help='Dimensionality of the input features.')
    parser.add_argument('--attention', type=utils.bool_flag, default=True,
                        help='Boolean flag indicating whether an Attention layer will be used.')
    parser.add_argument('--binarization', type=utils.bool_flag, default=False,
                        help='Boolean flag indicating whether a Binarization layer will be used.')
    parser.add_argument('--binary_bits', type=int, default=512,
                        help='Number of bits used in the Binarization layer. Applicable only when --binarization flag is true.')
    parser.add_argument('--f2f_sim_module', type=str, default='TopKChamfer', choices=['Chamfer', 'TopKChamfer', 'Max', 'Average'],
                        help='Frame-to-frame similarity calculation module.')
    parser.add_argument('--v2v_sim_module', type=str, default='TopKChamfer', choices=['Chamfer', 'TopKChamfer', 'Max', 'Average'],
                        help='Video-to-video similarity calculation module.')

    # losses
    parser.add_argument('--loss_selected', type=str, default='TBInnerQuadLinearAP,QuadLinearAP,InfoNCE,SSHN', help='Input a list of losses to use.')
    parser.add_argument('--qlap_sigma', default=0.1, type=float, help='the param sigma of the sigmoid used in QuadLinearAP loss')
    parser.add_argument('--qlap_rho', default=1, type=float, help='the param rho of the sigmoid used in QuadLinearAP loss')
    parser.add_argument('--innerAP_qlap_sigma', default=0.1, type=float, help='the param sigma of the sigmoid used in QuadLinearAP loss')
    parser.add_argument('--innerAP_qlap_rho', default=1, type=float, help='the param rho of the sigmoid used in QuadLinearAP loss')
    parser.add_argument('--inner_parameter', type=float, default=6., help='Parameter that determines the impact of the inner AP loss.')
    parser.add_argument('--f_topk_rate', default=0.3, type=float, help='the topk rate of spacial TopKChamfer')
    parser.add_argument('--v_topk_rate', default=0.1, type=float, help='the topk rate of temporal TopKChamfer')
    parser.add_argument('--pseudo_label_top_rate', default=0.3, type=float,   help='the rate of top K for pos pseudo labels')
    parser.add_argument('--pseudo_label_bottom_rate', default=0.3, type=float,   help='the rate of bottom K for neg pseudo labels')
    parser.add_argument('--use_pseudo_label', default=False, type=utils.bool_flag, help='whether to use DINO to generate pseudo_label')
    parser.add_argument('--pretrained_DINO_path', type=str, help='Path to the pretrained DINO.')
    parser.add_argument('--DINO_arch', type=str, default='vit_small', help='The arch of DINO.')
    parser.add_argument('--eval_step', type=int, default=400, help='Number of steps for evaluation.')

    # Training process
    parser.add_argument('--batch_sz', type=int, default=64,
                        help='Number of video pairs in each training batch.')
    parser.add_argument('--batch_sz_fe', type=int, default=512,
                        help='Number of frames in each batch for feature extraction.')
    parser.add_argument('--iter_epochs', type=int, default=30000,
                        help='Number of iterations to train the network.')
    parser.add_argument('--percentage', type=float, default=1.,
                        help='Dataset percentage used for training.')
    parser.add_argument('--learning_rate', type=float, default=4e-5,
                        help='Learning rate used during training.')
    parser.add_argument('--final_lr', type=float, default=1e-1,
                        help='Factor based on the the base lr used for the final learning rate for the lr scheduler.')
    parser.add_argument('--warmup_iters', type=int, default=1000,
                        help='Number of warmup iterations for the lr scheduler.')
    parser.add_argument('--warmup_lr_init', type=float, default=1e-2,
                        help='Factor based on the base lr used for the initial learning rate of warmup for the lr scheduler.')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay used during training.')
    parser.add_argument('--window_sz', type=int, default=32,
                        help='Number of frames of the loaded videos during training.')
    parser.add_argument('--temperature', type=float, default=0.03,
                        help='Temperature parameter for the infoNCE loss.')
    parser.add_argument('--lambda_parameter', type=float, default=3.,
                        help='Parameter that determines the impact of SSHN loss.')
    parser.add_argument('--r_parameter', type=float, default=1.,
                        help='Parameter that determines the impact of similarity regularization loss.')

    # Augmentations
    parser.add_argument('--augmentations', type=str, default='GT,FT,TT,ViV',
                        help='Transformations used for the strong augmentations. GT: Global Transformations '
                             'FT: Frame Transformations TT: Temporal Transformations ViV: Video-in-Video')
    parser.add_argument('--n_raug', type=int, default=2,
                        help='Number of augmentation transformations in RandAugment. Applicable when \'GT\' is in argument \'--augmentations\'')
    parser.add_argument('--m_raug', type=int, default=9,
                        help='Magnitude for all the transformations in RandAugment. Applicable when \'GT\' is in argument \'--augmentations\'')
    parser.add_argument('--p_overlay', type=float, default=.3,
                        help='Overlay probability in frame transformations. Applicable when \'FT\' is in argument \'--augmentations\'')
    parser.add_argument('--p_blur', type=float, default=.5,
                        help='Blur probability in frame transformations. Applicable when \'FT\' is in argument \'--augmentations\'')
    parser.add_argument('--p_tsd', type=float, default=.5,
                        help='Temporal Shuffle-Dropout probability in temporal transformations. Applicable when \'TT\' is in argument \'--augmentations\'')
    parser.add_argument('--p_ff', type=float, default=.1,
                        help='Fast Forward probability in temporal transformations. Applicable when \'TT\' is in argument \'--augmentations\'')
    parser.add_argument('--p_sm', type=float, default=.1,
                        help='Slow Motion probability in temporal transformations. Applicable when \'TT\' is in argument \'--augmentations\'')
    parser.add_argument('--p_rev', type=float, default=.1,
                        help='Revision probability in temporal transformations. Applicable when \'TT\' is in argument \'--augmentations\'')
    parser.add_argument('--p_pau', type=float, default=.1,
                        help='Pause probability in temporal transformations. Applicable when \'TT\' is in argument \'--augmentations\'')
    parser.add_argument('--p_shuffle', type=float, default=.5,
                        help='Shuffle probability in TSD. Applicable when \'TT\' is in argument \'--augmentations\'')
    parser.add_argument('--p_dropout', type=float, default=.3,
                        help='Dropout probability in TSD. Applicable when \'TT\' is in argument \'--augmentations\'')
    parser.add_argument('--p_content', type=float, default=.5,
                        help='Content probability in TSD. Applicable when \'TT\' is in argument \'--augmentations\'')
    parser.add_argument('--p_viv', type=float, default=.3,
                        help='Probability of applying video-in-video transformation. Applicable when \'ViV\' is in argument \'--augmentations\'')
    parser.add_argument('--lambda_viv', type=lambda x: tuple(map(float, x.split(','))), default=(.3, .7),
                        help='Resize factor range in video-in-video transformation. Applicable when \'ViV\' is in argument \'--augmentations\'')
    args = parser.parse_args()

    network_details = '{}_{}_D{}'.format(args.similarity_network.lower(), args.backbone, args.dims)
    network_details += '_att' if args.attention else ''
    network_details += '_bin_{}'.format(args.binary_bits) if args.binarization else ''

    args.loss_allowed = ['QuadLinearAP', 'TBInnerQuadLinearAP', 'InfoNCE', 'SSHN']
    
    if 'TBInnerQuadLinearAP' in args.loss_selected:
        args.use_pseudo_label = True 

    args.loss_list = args.loss_selected.split(',')
    
    timestamp = datetime.datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")

    training_details = '/HAPVR_loss_{}/it{}_ws{}_bs{}_f2f{}_v2v{}_sigma{}_rho{}_isigma{}_irho{}_topr{}_botr{}_fkr{}_vkr{}_ip{}_lr{}_wd{}/{}'.format(
        args.loss_selected, 
        args.iter_epochs, args.window_sz, args.batch_sz, 
        args.f2f_sim_module, args.v2v_sim_module, 
        args.qlap_sigma, args.qlap_rho,
        args.innerAP_qlap_sigma, args.innerAP_qlap_rho,
        args.pseudo_label_top_rate, args.pseudo_label_bottom_rate, 
        args.f_topk_rate, args.v_topk_rate,
        args.inner_parameter,
        args.learning_rate, args.weight_decay,
        timestamp)
   
    args.experiment_path += network_details + training_details
    main(args)
