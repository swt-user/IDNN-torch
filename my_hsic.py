import torch

def kernel_matrix(x):
    x1 = torch.unsqueeze(x, 0)
    x2 = torch.unsqueeze(x, 1)
    x3 = torch.pow(x1-x2, 2)
    x4 = torch.sum(x3, 2)
    return torch.exp( -0.5 * x4 )
    
def HSIC(Kx, Ky):
    m = Kx.size()[0]
    Kxy = torch.matmul(Kx, Ky)
    h  = torch.trace(Kxy) / m**2 + torch.mean(Kx)*torch.mean(Ky) - 2 * torch.mean(Kxy) / m
    return h * (m / (m-1))**2 

def compute_relative(x: torch.tensor, y:torch.tensor):
    """compute HSIC between x and y
    """
    Kx = kernel_matrix(x)
    Ky = kernel_matrix(y)
    return HSIC(Kx, Ky).item()
