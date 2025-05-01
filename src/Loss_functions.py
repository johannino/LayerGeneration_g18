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


def total_variation_loss(img):
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w

def gradient_penalty(discriminator, real_data, fake_data, condition):
    alpha = torch.rand(real_data.size(0), 1, 1, 1, device=real_data.device)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)

    d_interpolates, _, _ = discriminator(interpolates, condition)
    grad_outputs = torch.ones_like(d_interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
