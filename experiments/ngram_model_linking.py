#!/usr/bin/env python3

import os, pickle, collections, argparse, pathlib

import jax
import numpy as np

from kebab import mskebab
from kebab.utils.io_helpers import ItemJsonlReader
from kebab.contracts.entity import Entity
from disk.data.linking import LinkingCandidate
from diffusion_linking.surrogate_models import CountDict, get_ngram_counts
from DiSK_utils import ListWrapper, NameBigram

parser = argparse.ArgumentParser()
parser.add_argument('-prior_scale', type=float, default=0.01)
args = parser.parse_args()


benchmark = mskebab.Benchmark('data/sampled_rebel_clustering.json')
task = benchmark.tasks_by_name['Clustering-Heldout-Small']
data = ListWrapper([fragment[0] for fragment in task.read_items() ])
    

with open('data/wikipedia_names_2gram_counts.pickle', 'rb') as handle:
    count_dict = pickle.load(handle)
prior_counts = CountDict(count_dict['<UNK>'], count_dict)

surrogate = NameBigram(args.prior_scale, prior_counts)

  
def compute_lls(entities):
    '''
    Args:
        entities (list[Entity]): The entities.
    Returns:
        np.ndarray: The log probabilities as a 1-d numpy array.
    '''
    return np.array([surrogate.evidence(None, get_ngram_counts(entity.properties['name'], surrogate.n)) for entity in entities]).flatten()

print(compute_lls(data[0:10]))

def compute_linking_scores(candidates):
    '''
    Args:
        candidates (list[LinkingCandidate]): The pairs of linking candidates.
    Returns:
        np.ndarray: The linking scores as a 1-d numpy array.
    '''
    entities = [c.entity1 for c in candidates]
    entities += [c.entity2 for c in candidates]
    entities += [Entity.merge([c.entity1, c.entity2]) for c in candidates]
    
    log_probs = compute_lls(entities)    
    log_probs_1, log_probs_2, log_probs_merged = np.array_split(log_probs, 3)
    linking_scores = log_probs_merged - (log_probs_1 + log_probs_2)
    return linking_scores

candidates = [LinkingCandidate(data[0][0], data[6][0], None),
              LinkingCandidate(data[0][0], data[1][0], None),
              LinkingCandidate(data[3][0], data[5][0], None),
              LinkingCandidate(data[8][0], data[10][0], None),
              LinkingCandidate(data[18][0], data[19][0], None),
              LinkingCandidate(data[12][0], data[14][0], None),
              LinkingCandidate(data[14][0], data[10][0], None)
              ]
scores = compute_linking_scores(candidates)
print('\n'.join([f'{candidates[i].entity1.properties["name"]}, {candidates[i].entity2.properties["name"]}: {scores[i]}' for i in range(len(scores))]))