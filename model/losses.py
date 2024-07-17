import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_select(loss_name, args):
    if loss_name == 'TBInnerQuadLinearAP':
        loss_params = {'sigma': args.innerAP_qlap_sigma, 
                       'rho': args.innerAP_qlap_rho,
                       'batch_size': args.batch_sz*2,
                       'frame_num': args.window_sz}
        criterion = TBInnerQuadLinearAP(**loss_params)
    elif loss_name == 'QuadLinearAP':
        loss_params = {'sigma': args.qlap_sigma, 
                       'rho': args.qlap_rho,
                       'batch_size': args.batch_sz*2}
        criterion = QuadLinearAPLoss(**loss_params)
    elif loss_name == 'InfoNCE':
        loss_params = {'temperature': args.temperature}
        criterion = InfoNCELoss(**loss_params)
    elif loss_name == 'SSHN':
        loss_params = {}
        criterion = SSHNLoss(**loss_params)

    else:
        raise Exception('Loss {} not available!'.format(loss_name))

    return criterion



def heaviside(tensor):
    output = torch.where(tensor > 0, torch.ones_like(tensor, device=tensor.device), torch.zeros_like(tensor, device=tensor.device))
    return output


def quad_linear(tensor, sigma=0.1):
    output = torch.where(tensor > 0, 2*tensor/sigma+1,
                       torch.where((tensor >= -sigma) & (tensor <= 0), (tensor/sigma)**2+2*tensor/sigma+1, 
                                   torch.zeros_like(tensor)))
    return output


def compute_aff(x):
    return torch.mm(x, x.t())


class QuadLinearAPLoss(torch.nn.Module):
    def __init__(self, sigma, rho, batch_size):
        super(QuadLinearAPLoss, self).__init__()

        self.batch_size = batch_size
        self.sigma = sigma
        self.rho = rho

    def forward(self, sim_matrix, label_matrix):
        # Forward pass for all input predictions: 
        mask = 1.0 - torch.eye(self.batch_size, device=sim_matrix.device)
        mask = mask.unsqueeze(dim=0).repeat(self.batch_size, 1, 1)

        # compute the relevance scores via cosine similarity of the CNN-produced embedding vectors
        sim_all = sim_matrix
        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, self.batch_size, 1)
        # compute the difference matrix
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
        # pass through the quad linear function for all
        sim_sl = quad_linear(sim_diff, self.sigma) * mask
        # generate positive label matrix
        label_repeat = label_matrix.unsqueeze(dim=1).repeat(1, self.batch_size, 1)
        pos_label_mask = label_repeat * label_repeat.permute(0, 2, 1)
        # pass through the heaviside function only for pos pairs
        sim_sl_sg = torch.where(pos_label_mask == 1.0, self.rho*heaviside(sim_diff), sim_sl) 
        sim_sl_sg = sim_sl_sg * mask
        # compute the rankings
        sim_all_rk = torch.sum(sim_sl_sg, dim=-1) + 1

        # ------ differentiable ranking of only positive set in retrieval set ------
        sim_pos_sg = torch.where(pos_label_mask == 1, sim_sl_sg, torch.zeros_like(sim_sl_sg, device=sim_sl_sg.device))
        sim_pos_rk = torch.sum(sim_pos_sg, dim=-1) + 1
        # sum the values of the AP for all instances in the mini-batch for each row in sim_all_rk and sim_pos_rk
        pos_divide_vec = torch.sum((sim_pos_rk / sim_all_rk) * label_matrix, dim=-1)
        pos_cnt_vec = torch.sum(label_matrix, dim=-1)
        # get average AP value for each query
        ap = torch.mean(pos_divide_vec / pos_cnt_vec)

        return (1-ap)


class TBInnerQuadLinearAP(torch.nn.Module):
    def __init__(self, sigma, rho, batch_size, frame_num):
        super(TBInnerQuadLinearAP, self).__init__()

        self.batch_size = batch_size
        self.frame_num = frame_num
        self.sigma = sigma
        self.rho = rho

    def forward(self, sim_matrix, pseudo_pos_label_matrix, pseudo_neg_label_matrix, label_matrix):
        # Forward pass for all input predictions: 
        # ------ differentiable ranking of all retrieval set ------
        # compute the mask which ignores the relevance score of the query to itself
        mask = 1.0 - torch.eye(self.frame_num, device=sim_matrix.device)
        mask = mask.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).repeat(
            self.batch_size, self.batch_size, self.frame_num, 1, 1)
        sim_all = sim_matrix
        sim_all_repeat = sim_all.unsqueeze(dim=3).repeat(1, 1, 1, self.frame_num, 1)

        # compute the difference matrix
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 1, 2, 4, 3)
        # generate label matrix for pos-pos pairs
        pseudo_pos_label_repeat = pseudo_pos_label_matrix.unsqueeze(dim=3).repeat(1, 1, 1, self.frame_num, 1)
        pseudo_pos_pos_label_mask = pseudo_pos_label_repeat * pseudo_pos_label_repeat.permute(0, 1, 2, 4, 3)

        # generate label matrix for neg-pos pairs
        pseudo_neg_label_repeat = pseudo_neg_label_matrix.unsqueeze(dim=3).repeat(1, 1, 1, self.frame_num, 1)
        pseudo_pos_neg_label_mask = pseudo_neg_label_repeat * pseudo_pos_label_repeat.permute(0, 1, 2, 4, 3)
        # first pass through the quad linear function for all pos-neg pairs
        sim_sl = torch.where(pseudo_pos_neg_label_mask == 1.0, quad_linear(sim_diff, self.sigma), torch.zeros_like(sim_diff, device=sim_diff.device))
        sim_sl = sim_sl * mask
        # then pass through the heaviside/sigmoid pos function only for pos pairs (provided by pseudo_pos_pos_label_mask)
        # it doesn't metter whether other position use quad linear or heaviside, because other elements will not be used
        sim_sl_sg = torch.where(pseudo_pos_pos_label_mask == 1.0, self.rho*heaviside(sim_diff), sim_sl)
        sim_sl_sg = sim_sl_sg * mask
        # compute the rankings
        sim_all_rk = torch.sum(sim_sl_sg, dim=-1) + 1

        # ------ differentiable ranking of only positive set in retrieval set ------
        sim_pos_sg = torch.where(pseudo_pos_pos_label_mask == 1, sim_sl_sg, torch.zeros_like(sim_sl_sg, device=sim_sl_sg.device))
        # compute the rankings of the positive set
        sim_pos_rk = torch.sum(sim_pos_sg, dim=-1) + 1
        # sum the values of the SigLinearAP for all instances in the mini-batch
        # for each row in sim_all_rk and sim_pos_rk
        pos_divide_vec = torch.sum((sim_pos_rk / sim_all_rk) * pseudo_pos_label_matrix, dim=-1)
        pos_cnt_vec = torch.sum(pseudo_pos_label_matrix, dim=-1)
        # get average AP value for each query
        ap_matrix = torch.mean(pos_divide_vec / pos_cnt_vec, dim=-1)

        # only use the AP of positive video instance
        pos_ap_matrix = ap_matrix * label_matrix
        ap = torch.sum(pos_ap_matrix) / torch.sum(label_matrix)

        return (1-ap)



class InfoNCELoss(nn.Module):

    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, similarity, labels):
        assert similarity.shape == labels.shape
        # Get all negatives pairs
        identity = torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)
        non_matches = labels == 0
        nontrivial_matches = labels * (~identity)

        logits = (similarity / self.temperature).exp()
        partitions = logits + ((non_matches * logits).sum(dim=1) + 1e-6).unsqueeze(1)
        probabilities = logits / partitions
        infonce_loss = (
                (-probabilities.log() * nontrivial_matches).sum(dim=1)
                / nontrivial_matches.sum(dim=1)
        ).mean()
        return infonce_loss

    def __repr__(self, ):
        return '{}(temperature={})'.format(self.__class__.__name__, self.temperature)


class SSHNLoss(nn.Module):

    def __init__(self, eps=1e-3):
        super(SSHNLoss, self).__init__()
        self.eps = eps

    def forward(self, similarity, labels):
        assert similarity.shape == labels.shape
        # Get the hardest negative value for each video in the batch
        non_matches = labels == 0
        small_value = torch.tensor(-100.0).to(similarity)
        max_non_match_sim, _ = torch.where(non_matches, similarity, small_value).max(
            dim=1, keepdim=True
        )
        hardneg_loss = -(1-max_non_match_sim).clamp(min=self.eps).log().mean()
        self_loss = -torch.diagonal(similarity).clamp(min=self.eps).log().mean()

        return hardneg_loss, self_loss

    def __repr__(self, ):
        return '{}()'.format(self.__class__.__name__)


class SimilarityRegularizationLoss(nn.Module):

    def __init__(self, min_val=-1., max_val=1., reduction='sum'):
        super(SimilarityRegularizationLoss, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        assert reduction in ['sum', 'mean'], 'Invalid reduction value'
        self.reduction = reduction

    def forward(self, similarity):
        loss = torch.sum(torch.abs(torch.clamp(similarity - self.min_val, max=0.)))
        loss += torch.sum(torch.abs(torch.clamp(similarity - self.max_val, min=0.)))
        if self.reduction == 'mean':
            loss = loss / similarity.numel()
        return loss

    def __repr__(self, ):
        return '{}(min_val={}, max_val={})'.format(self.__class__.__name__, self.min_val, self.max_val)


