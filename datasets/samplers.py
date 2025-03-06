import math
import pandas as pd
import numpy as np

from copy import deepcopy
from itertools import chain

import torch

class Conformer_Batch_Sampler(torch.utils.data.sampler.Sampler):
    # groups ALL conformers of the same ligand (in data_source) into the same batch, regardless of the number of conformers
    # it may be a better option to set a maximum number of conformers per ligand to be grouped together
    
    def __init__(self, data_source, batch_size, groupby = 'Name_int', shuffle = True):
        self.data_source = data_source
        self.data_source['batch_index'] = np.arange(0, len(self.data_source))
        self.batch_size = batch_size
        self.groups = self.data_source.groupby([groupby], sort = False)
        self.shuffle = shuffle

    def __iter__(self): # returns list of lists of indices, with each inner list containing a batch of indices
        group_indices = [list(g[1].batch_index) for g in self.groups]
        
        if self.shuffle:
            np.random.shuffle(group_indices)

        batches = [list(chain(*group_indices[self.batch_size*i:self.batch_size*i+self.batch_size])) for i in range(math.ceil(len(self.groups)/self.batch_size))]
        return iter(batches)

    def __len__(self): # number of batches
        return math.ceil(len(self.groups) / self.batch_size) # includes last batch