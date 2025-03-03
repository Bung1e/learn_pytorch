import torch
from random import randint

def act(z):
    return torch.tanh(z)

def df(z):
    s = act(z)
    return 1 - s * s

def go_forward(x_inp, w1, w2):
    z1 = torch.mv(w1[:, :3], x_inp) + w1[:, 3]
    s1 = act(z1)

    z2 = torch.dot(w2[:2], s1) + w2[2]
    y = act(z2)

    return y, z1, s1, z2

def main():
    torch.manual_seed(1)
    
    W1 = torch.rand(8).view(2, 4) - 0.5
    W2 = torch.rand(3) - 0.5

    x_train = torch.FloatTensor([(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                            (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)])
    y_train = torch.FloatTensor([-1, 1, -1, 1, -1, 1, -1, -1])

    lmd = 0.05  
    N = 1000 
    total = len(y_train) 
    for _ in range(N):
        k = randint(0, total - 1)
        x = x_train[k]
        y, z1, s1, z_out = go_forward(x, W1, W2)

        e2 = y - y_train[k]
        delta2 = e2 * df(z_out)

        e1 = delta2 * W2[:2]
        delta1 = e1 * df(z1)

        W2[:2] -= lmd * delta2 * s1
        W2[2] -= lmd * delta2

        W1[0, :3] -= lmd * delta1[0] * x
        W1[1, :3] -= lmd * delta1[1] * x
        
        W1[0, 3] -= lmd * delta1[0]
        W1[1, 3] -= lmd * delta1[1]

    for x, d in zip(x_train, y_train):
        y, z1, s1, z_out = go_forward(x, W1, W2)
        print(f"out: {y} => {d}")

    print(W1, W2)




if __name__ == "__main__":
    main()