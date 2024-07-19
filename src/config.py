
class Config:
    n_tasks = 8 # 2^1 to 2^17

    train_steps = 100
    batch_size = 256
    learning_rate = 1e-3
    weight_decay = 1e-2

    n_points = 16
    dim = 8
    noise_scale = 0.5
