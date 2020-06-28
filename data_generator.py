import math
import operator
import threading

import numpy as np
import tensorflow as tf

from augment import get_policies, add_augment


class Image_Generator(tf.keras.utils.Sequence):

    def __init__(
        self,
        x_set,
        y_set,
        batch_size,
        policy_list
        ):

        self.x_set = x_set
        self.y_set = y_set
        self.batch_size = batch_size
        self.policies = [get_policies(policy) for policy in policy_list]
        self._set_probs()

    def __len__(self):
        return math.ceil(len(self.x_set) / self.batch_size)

    def __getitem__(self, step_idx):
        
        y_batch = self.y_set[self.batch_size * step_idx : self.batch_size * (step_idx + 1)]

        new_policies = []
        for p_idx in range(len(self.policies)):
            choose = np.random.choice(range(len(self.policies[p_idx])), len(y_batch), p=self.probs[p_idx])
            new_policies.append(operator.itemgetter(*choose)(self.policies[p_idx]))

        x_batch = []
        for x_idx, idx in enumerate(range(self.batch_size * step_idx, self.batch_size * (step_idx + 1) if self.batch_size * (step_idx + 1) < len(self.x_set) else len(self.x_set))):
            x = np.copy(self.x_set[idx])
            new_policy = [p[x_idx] for p in new_policies]
            x = add_augment(x, new_policy)
            x = np.asarray(x, dtype=np.float32) / 255.0
            x_batch.append(x)

        x_batch = np.asarray(x_batch)
        y_batch = np.asarray(y_batch)

        return x_batch, y_batch
        
    def on_epoch_end(self):
        self._shuffle()

    def _set_probs(self):
        self.probs = []
        for policy in self.policies:
            self.probs.append([p.get('prob') for p in policy])

    def _shuffle(self):
        shuffle_list = np.arange(self.x_set.shape[0])
        np.random.shuffle(shuffle_list)
        self.x_set = self.x_set[shuffle_list]
        self.y_set = self.y_set[shuffle_list]
        