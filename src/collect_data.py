import numpy as np
import torch

from config import Config
import utils as u

class LinRegData:
    def __init__(self, config, task_pool=None):
        self.config = config
        self.train_data = []
        if task_pool is not None:
            self.task_pool = task_pool
        else:
            self.generate_task_pool()

    def generate_task_pool(self):
        self.task_pool = np.random.normal(size=(self.config.n_tasks, self.config.dim))

    def generate_batch(self):
        batch_task_idx = np.random.randint(self.task_pool.shape[0], size=(self.config.batch_size))
        batch_tasks = self.task_pool[batch_task_idx,:]
        data = np.random.normal(size=(self.config.batch_size, self.config.n_points, self.config.dim))
        targets = data @ batch_tasks[:,:,np.newaxis] + np.random.normal(size=(self.config.batch_size, self.config.n_points, 1)) * self.config.noise_scale
        targets = np.squeeze(targets, axis=2)

        xs = torch.from_numpy(data).float()
        ys = torch.from_numpy(targets).float()
        ws = torch.from_numpy(batch_tasks).float()
        seq = u.to_seq(xs, ys)

        return {"ws": ws, "xs": xs, "ys": ys, "seq": seq}

    def batch_generator(self, n_batches):
        for _ in range(n_batches):
            yield self.generate_batch()

    def batch_iterator(self):
        return BatchIterator(generator=self.batch_generator(self.config.train_steps), length=self.config.train_steps)

class BatchIterator(torch.utils.data.IterableDataset):
    def __init__(self, generator, length=None):
        self.length = length
        self.generator = generator

    def __iter__(self):
        return self.generator
    
    def __len__(self):
        return self.length

if __name__ == "__main__":
    pass
