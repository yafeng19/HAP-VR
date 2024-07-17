import torch
import utils
import argparse
import torch.nn as nn
from collections import defaultdict, OrderedDict
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import EvaluationDataset
from model.similarity_network import SimilarityNetwork
from model.feature_extractor import FeatureExtractor
from datasets.generators import VideoDatasetGenerator, HDF5DatasetGenerator



@torch.no_grad()
def calculate_similarities_to_queries(model, queries, target):
    similarities = []
    batch_sz = 2048
    for i, query in enumerate(queries):
        if query.device.type == 'cpu':
            query = query.cuda()
        sim = []
        for batch in utils.batching(target, batch_sz):
            sim.append(model.calculate_video_similarity(query, batch, apply_visil=True))
        sim = torch.mean(torch.cat(sim, 0))
        similarities.append(sim.cpu().numpy())
    return similarities 
    
    
@torch.no_grad()
def query_vs_target(sim_network, dataset, query_loader, refer_loader, verbose=True):

    # Extract features of the queries
    all_db, queries, queries_ids = set(), [], []
    pbar = tqdm(query_loader) if verbose else query_loader
    for (video_tensor,), (video_id,) in pbar:
        if video_id and video_tensor.shape[0]:
            features = video_tensor.cuda()
            features = sim_network.index_video(features)
            all_db.add(video_id)
            queries.append(features)
            queries_ids.append(video_id)
            if verbose:
                pbar.set_postfix(query=video_id, features=features.shape)

    # Calculate similarities between the queries and the database videos
    similarities = dict({query: dict() for query in queries_ids})
    pbar = tqdm(refer_loader) if verbose else refer_loader
    for (video_tensor,), (video_id,) in pbar:
        if video_id and video_tensor.shape[0]:
            features = video_tensor.cuda()
            features = sim_network.index_video(features)
            sims = calculate_similarities_to_queries(sim_network, queries, features)
            all_db.add(video_id)
            for i, s in enumerate(sims):
                similarities[queries_ids[i]][video_id] = float(s)
            if verbose:
                pbar.set_postfix(target=video_id, features=features.shape)

    return dataset.evaluate(similarities, all_db, verbose=verbose)


def eval_on_FIVR(model_path, eval_dataset, query_loader, refer_loader):

    d = torch.load(model_path)
    sim_network = SimilarityNetwork[d['args'].similarity_network].get_model(**vars(d['args']))
    sim_network.load_state_dict(d['model'])
    sim_network = sim_network.cuda().eval()

    results = query_vs_target(sim_network, eval_dataset, query_loader, refer_loader)
    tasks = {'retrieval': OrderedDict(
                {'DSVR': ['ND', 'DS'], 'CSVR': ['ND', 'DS', 'CS'], 'ISVR': ['ND', 'DS', 'CS', 'IS']}),
            'detection': OrderedDict(
                {'DSVD': ['ND', 'DS'], 'CSVD': ['ND', 'DS', 'CS'], 'ISVD': ['ND', 'DS', 'CS', 'IS']})}
    
    ret = {'retrieval': OrderedDict(),
            'detection': OrderedDict()}
    for task in tasks['retrieval']:
        ret['retrieval'][task] = results[task]
    for task in tasks['detection']:
        ret['detection'][task] = results[task]

    return ret


def get_eval_data():
    dataset_hdf5 = './video_data/FIVR_5K/features/fivr_5k.hdf5'

    dataset = EvaluationDataset['FIVR_5K'].get_dataset()

    query_generator = HDF5DatasetGenerator(dataset_hdf5, dataset.get_queries())
    query_loader = DataLoader(query_generator, num_workers=8)

    refer_generator = HDF5DatasetGenerator(dataset_hdf5, dataset.get_database())
    refer_loader = DataLoader(refer_generator, num_workers=8)

    return dataset, query_loader, refer_loader

