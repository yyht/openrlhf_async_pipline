

import numpy as np
import torch
import json
from collections import OrderedDict

class MoPPS(object):
    def __init__(self, alpha_prior, beta_prior, exp_ratio=1.0, target_success_rate=0.5, top_k_ratio=0.6):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.uuids = OrderedDict({})
        self.target_success_rate = target_success_rate
        self.top_k_ratio = top_k_ratio
        self.exp_ratio = exp_ratio

    def selec_fn(self, predicted_difficulity):
        idx2uuid = OrderedDict({})
        predicted_score = []
        for idx, key in enumerate(predicted_difficulity):
            idx2uuid[idx] = key
            predicted_score.append(predicted_difficulity[key])
        
        predicted_score = np.array(predicted_score)
        mse = np.power(predicted_score - self.target_success_rate, 2.0)
        topk_num = int(len(predicted_score)*self.top_k_ratio)
        top_k_indices = np.argsort(mse)[:topk_num]
        return top_k_indices

    def sample(self, uuids):
        predicted_difficulity = OrderedDict({})
        for uuid_str in uuids:
            if uuid_str not in self.uuids:
                self.uuids[uuid_str] = {
                    'alpha': self.alpha_prior,
                    'beta': self.beta_prior,
                    'alpha_0': self.alpha_prior,
                    'beta_0': self.beta_prior
                }
            
            gamma = np.random.beta(self.uuids[uuid_str]['alpha'], self.uuids[uuid_str]['beta'], size=None)
            predicted_difficulity[uuid_str] = gamma
        
        select_indices = self.selec_fn(predicted_difficulity)
        return select_indices

    def update(self, actual_difficulity):
        for uuid_str in actual_difficulity:
            s_t = sum(actual_difficulity[uuid_str])
            k_t = len(actual_difficulity[uuid_str]) - s_t

            self.uuids[uuid_str]['alpha'] = self.exp_ratio * self.uuids['uuid_str']['alpha'] + (1.0-self.exp_ratio) * self.uuids['uuid_str']['alpha_0'] + s_t
            self.uuids[uuid_str]['beta'] = self.exp_ratio * self.uuids['uuid_str']['beta'] + (1.0-self.exp_ratio) * self.uuids['uuid_str']['beta_0'] + k_t
        


