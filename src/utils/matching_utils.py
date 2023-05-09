import torch
import tqdm


def run_corr_matrix(
    model_a, model_b, epochs=1, norm: bool = True, loader=train_aug_loader
):
    """
    given two networks net0, net1 which each output a feature map of shape NxCxWxH this will reshape
    both outputs to (N*W*H)xC and then compute a CxC correlation matrix between the outputs of the two networks
    N = dataset size, C = # of individual feature maps, H, W = height and width of one feature map
    :param model_a:
    :param model_b:
    :param epochs:
    :param norm:
    :param loader:
    :return:
    """
    n = epochs * len(loader)
    mean0 = mean1 = std0 = std1 = None
    with torch.no_grad():
        model_a.eval()
        model_b.eval()
        for _ in range(epochs):
            for i, (images, _) in enumerate(tqdm(loader)):
                img_t = images.float().cuda()
                out_a = model_a(img_t)
                out_a = out_a.reshape(out_a.shape[0], out_a.shape[1], -1).permute(
                    0, 2, 1
                )
                out_a = out_a.reshape(-1, out_a.shape[2]).double()

                out_b = model_b(img_t)
                out_b = out_b.reshape(out_b.shape[0], out_b.shape[1], -1).permute(
                    0, 2, 1
                )
                out_b = out_b.reshape(-1, out_b.shape[2]).double()

                mean0_b = out_a.mean(dim=0)
                mean1_b = out_b.mean(dim=0)
                std0_b = out_a.std(dim=0)
                std1_b = out_b.std(dim=0)
                outer_b = (out_a.T @ out_b) / out_a.shape[0]

                if i == 0:
                    mean0 = torch.zeros_like(mean0_b)
                    mean1 = torch.zeros_like(mean1_b)
                    std0 = torch.zeros_like(std0_b)
                    std1 = torch.zeros_like(std1_b)
                    outer = torch.zeros_like(outer_b)
                mean0 += mean0_b / n
                mean1 += mean1_b / n
                std0 += std0_b / n
                std1 += std1_b / n
                outer += outer_b / n

    cov = outer - torch.outer(mean0, mean1)
    if norm:
        corr = cov / (torch.outer(std0, std1) + 1e-4)
        return corr
    else:
        return cov
