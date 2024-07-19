
class Config:
    n_tasks = 128
    n_validation_batches = 50

    train_steps = 2**19
    checkpoint_interval = 2**16
    batch_size = 256
    learning_rate = 1e-3
    weight_decay = 1e-2

    n_points = 16
    dim = 8
    noise_scale = 0.5
