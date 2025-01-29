from cfr import CFR
import sys
import os
import torch
import torch.autograd.profiler as profiler
import time
sys.path.append(os.getcwd())

if __name__ == "__main__":
    cfr_iters = 75
    mcc_iters = 30
    round_iters = 100
    res_size = 150
    
    # device = torch.device("cuda")
    device = torch.device("cpu")
    
    start_time = time.time()
    test_cfr = CFR(cfr_iters, mcc_iters, round_iters, res_size, device)
    test_cfr.cfr_training()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    for i, strategy_model in enumerate(test_cfr.strategy_memory):
        torch.save(strategy_model[-1].state_dict(), f"training_cristi/models/player_{i}_model.pth")
        print("Models saved successfully!")

