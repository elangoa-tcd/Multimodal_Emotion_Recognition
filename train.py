import os
import random
import numpy as np
from config import get_config
from data_loader import get_loader
from solver import Solver
# from solver_original import Solver
import torch
import warnings
from utils.run_info import save_run_info


warnings.filterwarnings("ignore")  # ignore the warning

if __name__ == '__main__':
     # Setting the config for each stage
    train_config = get_config(mode='train')
    dev_config = get_config(mode='dev')
    test_config = get_config(mode='test')

    # Seed every RNG from --seed (default 55654, the value used for all reported runs)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    torch.cuda.manual_seed_all(train_config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(train_config)
    save_run_info(train_config, os.path.join(train_config.checkpoint_dir, 'devBest',f'run_{train_config.name}.json'))

    # Creating pytorch dataloaders
    train_data_loader = get_loader(train_config, shuffle = True)
    dev_data_loader = get_loader(dev_config, shuffle = False)
    test_data_loader = get_loader(test_config, shuffle = False)

    # Solver is a wrapper for model traiing and testing
    # solver = Solver
    solver = Solver(train_config,  train_data_loader, dev_data_loader, test_data_loader, is_train=True) 

    # Build the model
    solver.build()

    # Train the model (test scores will be returned based on dev performance)
    solver.train()






















