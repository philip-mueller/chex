from functools import partial
from typing import Tuple, Union
from torch import FloatTensor, Tensor, nn
import torch
from torch import autocast

class SoftRoiPool(nn.Module):
    def forward(self, x: Tensor, box_params: Tensor, beta: Union[float, FloatTensor] = 2.) -> Tuple[Tensor, Tensor]:
        """
        :param x: Featue map of image (N x H x W x d)
        :param roi_params: Parameters of bounding boxes (N x Q x 4) for Q boxes.
            format: (x_c, y_c, w, h) each in the range of [0, 1] (relative to image size)
        :return roi_features, roi_maps
            - roi_features: Pooled features of each roi (N x Q x d)
            - roi_maps: Attention maps of each roi (N x Q x H x W)
        """
        N, H, W, d = x.shape

        # Compute kernel on sampling grid
        sampled_grid = get_sample_grid(H, W, device=x.device, dtype=x.dtype)  # (H x W x 2)
        roi_patch_map = separable_generalized_gaussian_pdf(box_params.to(dtype=x.dtype), sampled_grid, beta=beta)  # (N x Q x H x W)

        # with autocast(device_type='cuda', enabled=False):
        # Batched matrix multiplication and normalize
        roi_features = torch.einsum('nqhw,nhwd->nqd', roi_patch_map.float(), x.float())  # (N x Q x d)
        roi_features = roi_features / H * W

        return roi_features, roi_patch_map


def get_sample_grid(H: int, W: int, device, dtype) -> Tensor:
    # (H x W)
    y, x = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
                          torch.arange(W, device=device, dtype=dtype),
                          indexing='ij')
    # (H x W x 2)
    sampled_grid = torch.stack([x, y], dim=-1)
    # consider pixel centers instead of left-upper position
    sampled_grid += 0.5
    # normalize positions into range [0, 1]
    sampled_grid[:, :, 0] /= W
    sampled_grid[:, :, 1] /= H
    return sampled_grid


def generalized_gauss_1d_log_pdf(mu: Tensor, sigma: Tensor, sampled_grid: Tensor,
                                 beta: Union[float, FloatTensor]) -> Tensor:
    """
    :param mu: (N x K)
    :param sigma: (N x K)
    :param sampled_grid: Sampled points (P) where P is the number of sampled points of the Gaussian pdf
    :return (N x K x P)
    """
    assert len(sampled_grid.shape) == 1
    assert len(mu.shape) == 2
    assert len(sigma.shape) == 2

    if not isinstance(beta, (float, int)):
        assert isinstance(beta, Tensor)
        if beta.numel() > 1:
            assert beta.shape == mu.shape
            beta = beta[:, :, None]

    # (unnormalized) log pdf = -0.5*((x-mu)/sigma)^2
    # log_pdf = - (1 / beta) * (
    #     (sampled_grid[None, None] - mu[:, :, None]) / sigma[:, :, None]
    # ).pow(beta)
    log_pdf = -(
        (sampled_grid[None, None] - mu[:, :, None]).abs() / sigma[:, :, None]
    ).pow(beta)
    return log_pdf


def separable_generalized_gaussian_pdf(box_params: Tensor, sampled_grid: Tensor,
                                       beta: Union[float, FloatTensor]) -> Tensor:
    """
    :param box_params: (N x Q x 4)
    :param sampled_grid: (... x 2)
    :return: (N x Q x ...)
    """
    N, K, _ = box_params.shape
    *dims, _ = sampled_grid.shape
    sampled_grid = sampled_grid.view(-1, 2)  # (... x 2)
    mu = box_params[:, :, :2]  # (N x K x 2)
    sigma = box_params[:, :, 2:]  # (N x K x 2)
    # compute x and y Gaussian pdf's independently (in log-space and non-normalized)
    log_scores_x = generalized_gauss_1d_log_pdf(mu[..., 0], sigma[..., 0],
                                                sampled_grid[..., 0], beta)  # (N x K x ...)
    log_scores_y = generalized_gauss_1d_log_pdf(mu[..., 1], sigma[..., 1],
                                                sampled_grid[..., 1], beta)  # (N x K x ...)
    # combine them in log space (multiplication in prob space)
    log_scores = log_scores_x + log_scores_y  # (N x K x ...)

    # Normalize to max value = 1
    scores = torch.exp(log_scores)
    probs = scores / (scores.max(-1, keepdim=True).values + 1e-12)

    # Alternative: convert to probs by applying exp and then normalizing to sum equals 1 == softmax
    # probs = torch.softmax(log_scores, dim=-1)  # (N x K x ...)

    return probs.view(N, K, *dims)

