import numpy as np
import matplotlib.pyplot as plt
import torch
from config import Config
from model import GPT2
import utils as u
from collect_data import LinRegData
from predictions import oracle, ridge
from tqdm import tqdm

if __name__ == "__main__":
    assert torch.cuda.is_available()
    device = 'cuda'

    config = Config()

    
