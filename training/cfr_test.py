from cfr import CFR
import sys
import os
import torch
import torch.autograd.profiler as profiler
sys.path.append(os.getcwd())

if __name__ == "__main__":
    cfr_iters = 1
    mcc_iters = 20
    round_iters = 100
    res_size = 100
    test_cfr = CFR(cfr_iters, mcc_iters, round_iters, res_size)
    test_cfr.CFR_training()

    for i, strategy_model in enumerate(test_cfr.strategy_memory):
        torch.save(strategy_model[-1].state_dict(), f"models/player_{i}_model.pth")
        print("Models saved successfully!")
