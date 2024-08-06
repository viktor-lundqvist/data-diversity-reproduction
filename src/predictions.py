import numpy as np
import torch

def oracle(batch):
    targets = torch.matmul(batch["xs"], batch["ws"][:,:,np.newaxis])
    preds = torch.squeeze(targets, axis=2)
    return preds

def zero(batch):
    return torch.zeros_like(batch['ys'])

class Ridge:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, batch):
        data = batch["xs"]
        targets = batch["ys"]

        batch_size, n_points, _ = data.shape
        targets = torch.unsqueeze(targets, -1)  # batch_size x n_points x 1
        preds = [torch.zeros(batch_size).to(data.get_device())]
        preds.extend(
            [self.predict(data[:, :_i], targets[:, :_i], data[:, _i : _i + 1]) for _i in range(1, n_points)]
        )
        preds = torch.stack(preds, axis=1)
        return preds

    def predict(self, X, Y, test_x):
        _, _, n_dims = X.shape
        XT = X.transpose(2, 1)  # batch_size x n_dims x i
        XT_Y = XT @ Y  # batch_size x n_dims x 1, @ should be ok (batched matrix-vector product)
        ridge_matrix = torch.matmul(XT, X) + self.sigma**2 * torch.eye(n_dims).to(X.get_device())  # batch_size x n_dims x n_dims
        # batch_size x n_dims x 1
        ws = torch.linalg.solve(ridge_matrix.float(), XT_Y.float())
        pred = test_x @ ws  # @ should be ok (batched row times column)
        return pred[:, 0, 0]

class DiscreteMMSE:
    def __init__(self, task_pool, sigma):
        self.task_pool = task_pool
        self.sigma = sigma

    def __call__(self, batch):
        """
        Args:
            data: batch_size x n_points x n_dims (float)
            targets: batch_size x n_points (float)
        Return:
            batch_size x n_points (float) OR batch_size x 1 IF index is not None
        """
        data = batch["xs"]
        targets = batch["ys"]
        device = batch["xs"].get_device()

        _, n_points, _ = data.shape
        targets = torch.unsqueeze(targets, -1)  # batch_size x n_points x 1
        W = torch.from_numpy(self.task_pool.squeeze().T).float().to(device)  # n_dims x n_tasks  (maybe do squeeze and transpose in setup?)
        preds = [torch.matmul(data[:, 0], W.mean(axis=1))]  # batch_size
        preds.extend(
            [
                self.predict(data[:, :_i], targets[:, :_i], data[:, _i : _i + 1], W)
                for _i in range(1, n_points)
            ]
        )
        preds = torch.stack(preds, axis=1)  # batch_size x n_points
        return preds

    def predict(self, X, Y, test_x, W):
        """
        Args:
            X: batch_size x i x n_dims (float)
            Y: batch_size x i x 1 (float)
            test_x: batch_size x 1 x n_dims (float)
            W: n_dims x n_tasks (float)
            scale: (float)
        Return:
            batch_size (float)
        """
        # X @ W is batch_size x i x n_tasks, Y is batch_size x i x 1, so broadcasts to alpha being batch_size x n_tasks
        alpha = -torch.sum((Y - torch.matmul(X, W))**2, axis=1) / (2*self.sigma**2)
        # softmax is batch_size x n_tasks, W.T is n_tasks x n_dims, so w_mmse is batch_size x n_dims x 1
        w_mmse = torch.unsqueeze(torch.matmul(torch.nn.Softmax(dim=1)(alpha), W.T), -1).float()
        # test_x is batch_size x 1 x n_dims, so pred is batch_size x 1 x 1. NOTE: @ should be ok (batched row times column)
        pred = test_x @ w_mmse
        return pred[:, 0, 0]
