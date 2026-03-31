import numpy as np
import torch
from model.PUNO import *

if __name__ == "__main__":
    data = np.load('examples/d1b28579fde95f19e1873a3963e0d14/points.npy')
    sparse_pc = data
    num_input = 2048
    up_ratio = 16
    index = np.random.permutation(sparse_pc.shape[0])[0:num_input]
    sparse_pc = sparse_pc[index, :]
    sparse_pc = torch.from_numpy(sparse_pc).unsqueeze(0).cuda().float()
    puno = PUNO(up_ratio=up_ratio).cuda().eval()
    ckpt = torch.load('ckpt/model.pth')
    puno.load_state_dict(ckpt)
    with torch.no_grad():
        np.savetxt('test_output.xyz', puno(sparse_pc).squeeze(0).detach().cpu().numpy())