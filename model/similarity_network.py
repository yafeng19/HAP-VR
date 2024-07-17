import torch
import enum
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod
from einops import rearrange, repeat

from model.layers import Attention, BinarizationLayer, MLPALayer, NetVLAD
from model.losses import SimilarityRegularizationLoss
from model.similarities import TopKChamferSimilarity, ChamferSimilarity, MaxSimilarity, AverageSimilarity, VideoComparator


class SimilarityNetwork(enum.Enum):
    ViSiL = enum.auto()

    def get_model(self, **kwargs):
        return self._get_config(self, **kwargs)

    def _get_config(self, value, **kwargs):
        return {
            self.ViSiL: ViSiL(**kwargs),
        }[value]


class SimilarityNetworkABC(nn.Module):

    @abstractmethod
    def index_video(self, x, mask=None):
        pass

    @abstractmethod
    def calculate_video_similarity(self, query, target, **kargs):
        pass

    @staticmethod
    def check_dims(features, mask=None, axis=0):
        if features.ndim == 4:
            return features, mask
        elif features.ndim == 3:
            features = features.unsqueeze(axis)
            if mask is not None:
                mask = mask.unsqueeze(axis)
            return features, mask
        else:
            raise Exception('Wrong shape of input video tensor. The shape of the tensor must be either '
                            '[N, T, R, D] or [T, R, D], where N is the batch size, T the number of frames, '
                            'R the number of regions and D number of dimensions. '
                            'Input video tensor has shape {}'.format(features.shape))


class ViSiL(SimilarityNetworkABC):

    def __init__(self,
                 dims=512,
                 attention=False,
                 attention_norm=False,
                 attention_activation=torch.tanh,
                 mlp=False,
                 mlp_norm=True,
                 binarization=False,
                 binary_bits=512,
                 f2f_sim_module = 'TopKChamfer',
                 v2v_sim_module = 'TopKChamfer',
                 f_topk_rate = 0.1,
                 v_topk_rate = 0.1,
                 activation=nn.Hardtanh(),
                 symmetric=False,
                 batch_norm=False,
                 **kwargs
                 ):
        super(ViSiL, self).__init__()
        if attention and binarization:
            raise Exception('Can\'t use \'attention=True\' and \'binarization=True\' at the same time. '
                            'Select one of the two options.')
        elif binarization:
            self.idx_type = 'bin'
            self.binarization = BinarizationLayer(bits=binary_bits)
        elif attention:
            self.idx_type = 'att'
            self.attention = Attention(dims, norm=attention_norm, activation=attention_activation)
        else:
            self.idx_type = 'none'

        self.dims = dims

        self.f2f_sim_module = f2f_sim_module
        self.v2v_sim_module = v2v_sim_module

        if self.f2f_sim_module == 'Chamfer':
            self.f2f_sim = ChamferSimilarity(symmetric=symmetric, axes=[3, 2])
        elif self.f2f_sim_module == 'TopKChamfer':
            self.f2f_sim = TopKChamferSimilarity(symmetric=symmetric, axes=[3, 2], top_rate=f_topk_rate)
        elif self.f2f_sim_module == 'Max':
            self.f2f_sim = MaxSimilarity(axes=[3, 2])
        elif self.f2f_sim_module == 'Average':
            self.f2f_sim = AverageSimilarity(axes=[3, 2])
        
        self.comparator_module = VideoComparator(batch_norm=batch_norm)

        if self.v2v_sim_module == 'Chamfer':
            self.v2v_sim = ChamferSimilarity(symmetric=symmetric, axes=[3, 2])
        elif self.v2v_sim_module == 'TopKChamfer':
            self.v2v_sim = TopKChamferSimilarity(symmetric=symmetric, axes=[3, 2], top_rate=v_topk_rate)
        elif self.v2v_sim_module == 'Max':
            self.v2v_sim = MaxSimilarity(axes=[3, 2])
        elif self.v2v_sim_module == 'Average':
            self.v2v_sim = AverageSimilarity(axes=[3, 2])

        self.sim_criterion = SimilarityRegularizationLoss()
        self.activation = activation


    def frame_to_frame_similarity(self, query, target, query_mask=None, target_mask=None, batched=False):
        sim_mask = None
        if batched:
            sim = torch.einsum('biok,bjpk->biopj', query, target)
            sim = self.f2f_sim(sim)
            sim = rearrange(sim, 'b i j -> b () i j')
            if query_mask is not None and target_mask is not None:
                sim_mask = torch.einsum('bik,bjk->bij', query_mask.unsqueeze(-1), target_mask.unsqueeze(-1))
                sim_mask = rearrange(sim_mask, 'b i j -> b () i j')
        else:
            sim = torch.einsum('aiok,bjpk->aiopjb', query, target)
            sim = self.f2f_sim(sim)
            frame_level_sim = rearrange(sim, 'a i j b -> a b i j')
            sim = rearrange(sim, 'a i j b -> (a b) () i j')
            if query_mask is not None and target_mask is not None:
                sim_mask = torch.einsum('aik,bjk->aijb', query_mask.unsqueeze(-1), target_mask.unsqueeze(-1))
                frame_level_sim_mask = rearrange(sim_mask, 'a i j b -> a b i j')
                sim_mask = rearrange(sim_mask, 'a i j b -> (a b) () i j')
        if hasattr(self, 'binarization'):
            sim /= target.shape[-1]
        if sim_mask is not None:
            frame_level_sim = frame_level_sim.masked_fill((1 - frame_level_sim_mask).bool(), 0.0)
            sim = sim.masked_fill((1 - sim_mask).bool(), 0.0)
        return frame_level_sim, sim, sim_mask


    def calculate_video_similarity(self, query, target, 
                                   query_mask=None, target_mask=None, 
                                   apply_self_attention = False,
                                   apply_visil=True, pooling=True):
        query, query_mask = self.check_dims(query, query_mask)
        target, target_mask = self.check_dims(target, target_mask)

        v2v_sim, sim_mask = self.similarity_matrix(query, target, batched=False, query_mask=query_mask, target_mask=target_mask,
                                                   apply_self_attention=apply_self_attention, apply_visil=apply_visil, pooling=pooling)
        sim = self.v2v_sim(v2v_sim, sim_mask)
        return sim.view(query.shape[0], target.shape[0])

    def similarity_matrix(self, query, target, batched=False,
                          query_mask=None, target_mask=None,
                          apply_self_attention=False,
                          apply_visil=True, pooling=True,
                          normalize=False, return_f2f=False):

        query, query_mask = self.check_dims(query, query_mask)
        target, target_mask = self.check_dims(target, target_mask)

        frame_level_sim, f2f_sim, sim_mask = self.frame_to_frame_similarity(query, target, query_mask, target_mask, batched=batched)
        if not apply_visil:
            return f2f_sim, sim_mask

        sim, sim_mask = self.comparator_module(f2f_sim, sim_mask, pooling=pooling)
        sim = self.activation(sim)

        if normalize:
            sim = sim / 2. + 0.5

        if return_f2f:
            return sim, f2f_sim, sim_mask
        return sim, sim_mask

    def index_video(self, x, mask=None):
        if hasattr(self, 'attention'):
            x, a = self.attention(x)
        if hasattr(self, 'mlp'):
            x = self.mlp(x)
        if hasattr(self, 'binarization'):
            x = self.binarization(x)
        if mask is not None:
            m = rearrange((1 - mask).bool(), 'a i -> a i () ()')
            x = x.masked_fill(m, 0.0)
        return x


    def forward(self, x, masks=None):
        N = x.shape[0]
        x = self.index_video(x, masks)
        frame_level_sim, sim, sim_mask = self.frame_to_frame_similarity(
            x, x, masks, masks, batched=False)
        sim, sim_mask = self.comparator_module(sim, sim_mask)
        loss = self.sim_criterion(sim)
        sim = self.activation(sim) / 2. + 0.5
        sim = self.v2v_sim(sim, sim_mask)
        return frame_level_sim, sim.view(N, N), loss
