import torch

def color_histogram_loss(generated, target, bins=64):
    loss = 0
    for c in range(3):  
        gen_hist = torch.histc(generated[:, c, :, :], bins=bins, min=0, max=1)
        target_hist = torch.histc(target[:, c, :, :], bins=bins, min=0, max=1)
        
        gen_hist = gen_hist / gen_hist.sum()
        target_hist = target_hist / target_hist.sum()

        loss += torch.abs(torch.cumsum(gen_hist, dim=0) - torch.cumsum(target_hist, dim=0)).sum()
    
    return loss / 3.0  