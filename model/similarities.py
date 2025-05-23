import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from einops import rearrange


class ChamferSimilarity(nn.Module):

    def __init__(self, symmetric=False, axes=[1, 0]):
        super(ChamferSimilarity, self).__init__()
        self.axes = axes
        self.symmetric = symmetric

    @staticmethod
    def chamfer_similarity(s, mask=None, max_axis=1, mean_axis=0, keepdim=False):
        if mask is not None:
            s = s.masked_fill((1 - mask).bool(), -np.inf)
            s = torch.max(s, max_axis, keepdim=True)[0]
            mask = torch.max(mask, max_axis, keepdim=True)[0]
            s = s.masked_fill((1 - mask).bool(), 0.0)
            s = torch.sum(s, mean_axis, keepdim=True)
            s /= torch.sum(mask, mean_axis, keepdim=True)
        else:
            s = torch.max(s, max_axis, keepdim=True)[0]
            s = torch.mean(s, mean_axis, keepdim=True)
        if not keepdim:
            s = s.squeeze(max(max_axis, mean_axis)).squeeze(min(max_axis, mean_axis))
        return s

    def symmetric_chamfer_similarity(self, s, mask=None, keepdim=False):
        return (self.chamfer_similarity(s, mask=mask, max_axis=self.axes[0], mean_axis=self.axes[1], keepdim=keepdim) +
                self.chamfer_similarity(s, mask=mask, max_axis=self.axes[1], mean_axis=self.axes[0], keepdim=keepdim)) / 2
    
    def forward(self, s, mask=None, keepdim=False):
        if self.symmetric:
            return self.symmetric_chamfer_similarity(s, mask, keepdim)
        return self.chamfer_similarity(s, mask, max_axis=self.axes[0], mean_axis=self.axes[1], keepdim=keepdim)

    def __repr__(self,):
        return '{}(max_axis={}, mean_axis={})'.format(self.__class__.__name__, self.axes[0], self.axes[1])



class TopKChamferSimilarity(nn.Module):
    ''' 
        For each query frame, find top-k max query frame and calculate mean similarity,
        the calculate the average similarity of all query frame.
    '''
    def __init__(self, symmetric=False, axes=[1, 0], top_rate=0.3):
        super(TopKChamferSimilarity, self).__init__()
        self.axes = axes
        self.symmetric = symmetric
        self.top_rate = top_rate

    @staticmethod
    def topk_chamfer_similarity(s, mask=None, max_axis=1, mean_axis=0, top_rate=0.3, keepdim=False):
        if mask is not None:
            s = s.masked_fill((1 - mask).bool(), -np.inf)
            topk = max(round(s.shape[max_axis]*top_rate), 1)
            s = torch.topk(s, topk, max_axis)[0]
            s = torch.mean(s, max_axis, keepdim=True)
            mask = torch.max(mask, max_axis, keepdim=True)[0]
            s = s.masked_fill((1 - mask).bool(), 0.0)
            s = torch.sum(s, mean_axis, keepdim=True)
            s /= torch.sum(mask, mean_axis, keepdim=True)
        else:
            topk = max(round(s.shape[max_axis]*top_rate), 1)
            s = torch.topk(s, topk, max_axis)[0]
            s = torch.mean(s, max_axis, keepdim=True)
            s = torch.mean(s, mean_axis, keepdim=True)
        if not keepdim:
            s = s.squeeze(max(max_axis, mean_axis)).squeeze(min(max_axis, mean_axis))
        return s

    def symmetric_topk_chamfer_similarity(self, s, mask=None, keepdim=False):
        return (self.topk_chamfer_similarity(s, mask=mask, max_axis=self.axes[0], mean_axis=self.axes[1], keepdim=keepdim) +
                self.topk_chamfer_similarity(s, mask=mask, max_axis=self.axes[1], mean_axis=self.axes[0], keepdim=keepdim)) / 2
    
    def forward(self, s, mask=None, keepdim=False):
        if self.symmetric:
            return self.symmetric_topk_chamfer_similarity(s, mask, keepdim)
        return self.topk_chamfer_similarity(s, mask, max_axis=self.axes[0], mean_axis=self.axes[1], top_rate=self.top_rate, keepdim=keepdim)

    def __repr__(self,):
        return '{}(max_axis={}, mean_axis={})'.format(self.__class__.__name__, self.axes[0], self.axes[1])


class MaxSimilarity(nn.Module):

    def __init__(self, axes=[1, 0]):
        super(MaxSimilarity, self).__init__()
        self.axes = axes

    def mac(self, x, mask=None, axis_1=1, axis_2=0, keepdim=False):
        x = torch.max(x, axis_1, keepdim=True)[0]
        x = torch.max(x, axis_2, keepdim=True)[0]
        if not keepdim:
            x = x.squeeze(max(axis_1, axis_2)).squeeze(min(axis_1, axis_2))
        return x

    def forward(self, sim, mask=None, keepdim=False):
        return self.mac(sim, mask, axis_1=self.axes[0], axis_2=self.axes[1], keepdim=keepdim)
        
    def __repr__(self):
        return self.__class__.__name__ + '()'


class AverageSimilarity(nn.Module):

    def __init__(self, axes=[1, 0]):
        super(AverageSimilarity, self).__init__()
        self.axes = axes
        
    def spoc(self, x, mask=None, axis_1=1, axis_2=0, keepdim=False):
        x = torch.mean(x, axis_1, keepdim=True)
        x = torch.mean(x, axis_2, keepdim=True)
        if not keepdim:
            x = x.squeeze(max(axis_1, axis_2)).squeeze(min(axis_1, axis_2))
        return x

    def forward(self, sim, mask=None, keepdim=False):
        return self.spoc(sim, mask, axis_1=self.axes[0], axis_2=self.axes[1], keepdim=keepdim)
        
    def __repr__(self):
        return self.__class__.__name__ + '()'


class VideoComparator(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, factor=1, batch_norm=False):
        super(VideoComparator, self).__init__()

        if batch_norm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, factor * 32, kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(factor * 32)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(factor * 32, factor * 64, kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(factor * 64)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(factor * 64, factor * 128, kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(factor * 128)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, factor * 32, kernel_size=(3, 3), padding=0)
            self.conv2 = nn.Conv2d(factor * 32, factor * 64, kernel_size=(3, 3), padding=0)
            self.conv3 = nn.Conv2d(factor * 64, factor * 128, kernel_size=(3, 3), padding=0)
        self.fconv = nn.Conv2d(factor * 128, out_channels, kernel_size=(1, 1))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, sim_matrix, mask=None, pooling=True):
        sim, mask = self._check_dims(sim_matrix, mask)
        sim = self._apply_mask(sim, mask)
        
        sim = self._padding(sim, pooling)
        sim = self.conv1(sim)
        sim = self._apply_mask(sim, mask)
        sim = F.relu(sim)
        sim, mask = self._pooling(sim, mask, pooling)
            
        sim = self._padding(sim, pooling)
        sim = self.conv2(sim)
        sim = self._apply_mask(sim, mask)
        sim = F.relu(sim)
        sim, mask = self._pooling(sim, mask, pooling)

        sim = self._padding(sim, pooling)
        sim = self.conv3(sim)
        sim = self._apply_mask(sim, mask)
        sim = F.relu(sim)

        sim = self.fconv(sim)
        sim = self._apply_mask(sim, mask)
        return sim, mask

    @staticmethod
    def _apply_mask(sim_matrix, mask):
        return sim_matrix.masked_fill((1 - mask).bool(), 0.0) if mask is not None else sim_matrix

    @staticmethod
    def _padding(sim_matrix, pooling):
        if pooling:
            s = F.pad(sim_matrix, (1, 1, 1, 1), 'constant', 0.)
        else:
            s = F.pad(sim_matrix, (1, 1, 1, 1), 'replicate')
        return s

    @staticmethod
    def _pooling(sim_matrix, mask, pooling):
        if pooling:
            sim_matrix = F.max_pool2d(sim_matrix, kernel_size=2, stride=2, padding=0)
            mask = F.max_pool2d(mask, kernel_size=2, stride=2, padding=0) if mask is not None else mask
        return sim_matrix, mask

    @staticmethod
    def _check_dims(sim_matrix, mask=None):
        if sim_matrix.ndim == 3:
            sim_matrix = sim_matrix.unsqueeze(1)
        elif sim_matrix.ndim != 4:
            raise Exception('Input Tensor to VideoComperator have to be 3D or 4D')

        if mask is not None:
            assert mask.shape[-2:] == sim_matrix.shape[-2:], \
                'Mask tensor must be of the same shape as similarity ' \
                'matrix in the last two dimensions. Mask shape is {} ' \
                'while similarity matrix is {}'.format(mask.shape[-2:], sim_matrix.shape[-2:])
        return sim_matrix, mask
