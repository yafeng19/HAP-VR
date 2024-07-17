import numpy as np
import pickle as pk

from collections import defaultdict, OrderedDict
from sklearn.metrics import average_precision_score

class SVD(object):

    def __init__(self, version='unlabeled'):
        self.name = 'SVD'
        self.ground_truth = self.load_groundtruth('data/test_groundtruth')
        self.unlabeled_keys = self.get_unlabeled_keys('data/unlabeled-data-id')
        if version == 'labeled':
            self.unlabeled_keys = []
        self.database = []
        for k, v in self.ground_truth.items():
            self.database.extend(list(map(str, v.keys())))
        self.queries = sorted(list(map(str, self.ground_truth.keys())))            
        self.database += self.unlabeled_keys
        self.database = sorted(self.database)

    def load_groundtruth(self, filepath):
        gnds = OrderedDict()
        with open(filepath, 'r') as fp:
            for idx, lines in enumerate(fp):
                tmps = lines.strip().split(' ')
                qid = tmps[0]
                cid = tmps[1]
                gt = int(tmps[-1])
                if qid not in gnds:
                    gnds[qid] = {cid: gt}
                else:
                    gnds[qid][cid] = gt
        return gnds

    def get_unlabeled_keys(self, filepath):
        videos = list()
        with open(filepath, 'r') as fp:
            for tmps in fp:
                videos.append(tmps.strip())
        return videos

    def get_queries(self):
        return self.queries

    def get_database(self):
        return self.database
    

    def calculate_metric(self, y_true, y_score, gt_len):
        y_true = np.array(y_true)[np.argsort(y_score)[::-1]]
        precisions = np.cumsum(y_true) / (np.arange(y_true.shape[0]) + 1)
        recall_deltas = y_true / gt_len
        return np.sum(precisions * recall_deltas)
    
    def calculate_uAP(self, similarities, all_db):
        y_true, y_score, gt_len = [], [], 0
        for query, targets in self.ground_truth.items():
            res = similarities[query]
            if isinstance(res, (np.ndarray, np.generic)):
                res = {v: s for v, s in zip(self.database, res) if v in all_db}
            for target, label in targets.items():
                if target in all_db:
                    s = res[target]
                    y_true.append(label)
                    y_score.append(s)

            for target in self.unlabeled_keys:
                if target in all_db:
                    s = res[target]
                    y_true.append(0)
                    y_score.append(s)

        gt_len = y_true.count(1)
        return self.calculate_metric(y_true, y_score, gt_len)


    def evaluate(self, similarities, all_db=None, verbose=True):
        mAP = []
        not_found = len(self.ground_truth.keys() - similarities.keys())
        uAP = self.calculate_uAP(similarities, all_db)
        for query, targets in self.ground_truth.items():
            y_true, y_score = [], []
            res = similarities[query]
            if isinstance(res, (np.ndarray, np.generic)):
                res = {v: s for v, s in zip(self.database, res) if v in all_db}
            for target, label in targets.items():
                if target in all_db:
                    s = res[target]
                    y_true.append(label)
                    y_score.append(s)

            for target in self.unlabeled_keys:
                if target in all_db:
                    s = res[target]
                    y_true.append(0)
                    y_score.append(s)
            mAP.append(average_precision_score(y_true, y_score))
        if verbose:
            print('=' * 5, 'SVD Dataset', '=' * 5)
            if not_found > 0:
                print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))
            print('Database: {} videos'.format(len(all_db)))

            print('-' * 16)
            print('mAP: {:.4f}'.format(np.mean(mAP)))
            print('uAP: {:.4f}'.format(uAP))
        return {'mAP': np.mean(mAP), 'uAP': uAP}