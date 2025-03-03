import torch
import numpy as np


def act(x):
    if x >= 0.5:
        return 1
    else:
        return 0

def go(house, rock, attr):
    x = torch.tensor([house, rock, attr], dtype=torch.float32)
    w_h = torch.tensor([[0.3, 0.3, 0], [0.4, -0.5, 1]])
    w_out = torch.tensor([-1, 1], dtype=torch.float32)
    
    z_h = torch.mv(w_h, x)
    u_h = torch.tensor([act(x) for x in z_h], dtype=torch.float32)

    z_out = torch.dot(u_h, w_out)
    answer = act(z_out)
    return answer

print(go(1, 1, 0))