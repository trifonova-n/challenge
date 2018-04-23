import numpy as np
import logging
from dataset import SkipSampleError


class DataLoader(object):
    def __init__(self, dataset, batch_size, shuffle=True, seed=1):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.seed = seed
        self.randomState = np.random.RandomState(seed=seed)

    def __iter__(self):
        if self.shuffle:
            indices = self.randomState.permutation(len(self.dataset))
        else:
            indices = range(len(self.dataset))
        sample = self.dataset[0]
        image_shape = sample['pixel_data'].shape
        batch_shape = (self.batch_size, image_shape[0], image_shape[1])
        batch = {}
        batch_index = 0
        for index in indices:
            try:
                sample = self.dataset[index]
            except SkipSampleError as e:
                logging.warning('Skipping', self.dataset.filenames[index], e)
                continue
            for key in sample:
                batch[key] = batch.get(key, np.zeros(batch_shape, sample[key].dtype))
                batch[key][batch_index, :, :] = sample[key]
            batch_index += 1
            if batch_index == self.batch_size:
                yield batch
                batch_index = 0
        # remaining data less than batch_size
        if batch_index > 0:
            for key in sample:
                batch[key] = batch[key][:batch_index, :, :]
            yield batch
