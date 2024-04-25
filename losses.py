from functools import cache
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure
import torch
import numpy as np
import skimage as ski
import scipy.ndimage as ndimage


def mse(generated, gt):
    loss = (generated[:, :3, :, :] - gt[:, :3, :, :]) ** 2
    loss = loss.sum(dim=1)
    loss = loss.sum() / (loss.shape[-1] * loss.shape[-2])
    return loss


def masked_mse(generated, gt):
    foreground_mask = (gt[:, -1, :, :] > 0).float()
    loss = (generated[:, :3, :, :] - gt[:, :3, :, :]) ** 2
    loss = loss.sum(dim=1)
    loss *= foreground_mask
    loss = loss.sum() / foreground_mask.float().sum()
    return loss


def masked_cosine(generated, gt, eps=1e-5):
    foreground_mask = (gt[:, -1, :, :] > 0).float()

    dot = (generated[:, :3, :, :] * gt[:, :3, :, :]).sum(dim=1)
    norm = generated[:, :3, :, :].norm(dim=1) * gt[:, :3, :, :].norm(dim=1)

    zero_mask = norm < eps
    dot *= (~zero_mask).type(dot.dtype)  # make sure gradients don't flow to elements considered zero
    norm[zero_mask] = 1  # avoid division by zero
    cosine_similarity = dot / norm
    loss = 1 - cosine_similarity

    loss *= foreground_mask
    loss = loss.sum() / foreground_mask.float().sum()
    return loss


def intensity(generated, gt):
    intensity_generated = generated[:, 3, :, :]
    intensity_gt = gt[:, 3, :, :]
    loss = (intensity_generated - intensity_gt).abs()
    loss = loss.sum() / (loss.shape[-1] * loss.shape[-2])
    return loss


def mse_intensity(generated, gt):
    intensity_generated = generated[:, 3, :, :]
    intensity_gt = gt[:, 3, :, :]
    loss = (intensity_generated - intensity_gt) ** 2
    loss = loss.sum() / (loss.shape[-1] * loss.shape[-2])
    return loss


def pixel_intensity(generated, gt):
    loss = (generated[:, :3, :, :].norm(dim=1) - gt[:, :3, :, :].norm(dim=1)).abs()
    loss = loss.sum() / (loss.shape[-1] * loss.shape[-2])
    return loss


def masked_pixel_intensity(generated, gt):
    foreground_mask = (gt[:, -1, :, :] > 0).float()
    loss = (generated[:, :3, :, :].norm(dim=1) - gt[:, :3, :, :].norm(dim=1)).abs() * foreground_mask
    loss = loss.sum() / foreground_mask.sum()
    return loss


def background(generated, gt):
    background_mask = (gt == 0)
    loss = generated[background_mask].sum(dim=1).mean()
    return loss



def SSIM(generated, gt):
    return 1 - multiscale_structural_similarity_index_measure(generated, gt, data_range =1.0)


def color_SSIM(generated, gt):
    return 1 - multiscale_structural_similarity_index_measure(generated[:, :3, :, :], gt[:, :3, :, :], data_range =1.0)


def weighted_cross_entropy_loss(generated, gt):
    gt = gt/255
    generated = torch.sigmoid(generated)
    img = torch.Tensor.cpu(gt).detach().numpy()
    weight = ndimage.gaussian_filter(img.astype(float), (0, 0, 1, 1)) + 4*ndimage.gaussian_filter(img.astype(float), (0, 0, 8, 8)) + 3*ndimage.gaussian_filter(img.astype(float), (0, 0, 16, 16)) + 2*ndimage.gaussian_filter(img.astype(float), (0, 0, 24, 24)) + ndimage.gaussian_filter(img.astype(float), (0, 0, 32, 32))
    weight = np.clip(weight/np.amax(weight-15), 0, 1)
    weight = np.maximum(img/255, weight)
    weight = torch.tensor(weight).float().to(generated.device)
    loss = - (1-weight) * (gt*torch.log(generated+1e-8) + (1-gt)*torch.log(1+1e-8-generated))
    loss = loss.sum(dim=1)
    loss = loss.sum() / (loss.shape[-1] * loss.shape[-2])
    return loss