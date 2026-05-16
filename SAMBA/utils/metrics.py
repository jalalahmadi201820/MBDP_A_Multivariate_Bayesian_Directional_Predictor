# -*- coding: utf-8 -*-
"""
Evaluation metrics for traffic prediction
"""

import torch
import numpy as np


def MAE_torch(pred, true, mask_value=None):
    """Mean Absolute Error"""
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred))


def MSE_torch(pred, true, mask_value=None):
    """Mean Squared Error"""
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((pred - true) ** 2)


def RMSE_torch(pred, true, mask_value=None):
    """Root Mean Squared Error"""
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))


def RRSE_torch(pred, true, mask_value=None):
    """Root Relative Squared Error"""
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.sum((pred - true) ** 2)) / torch.sqrt(torch.sum((pred - true.mean()) ** 2))


def MAPE_torch(pred, true, mask_value=None):
    """Mean Absolute Percentage Error"""
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))


def PNBI_torch(pred, true, mask_value=None):
    """Percentage of Normalized Bias"""
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    indicator = torch.gt(pred - true, 0).float()
    return indicator.mean()


def oPNBI_torch(pred, true, mask_value=None):
    """Overall Percentage of Normalized Bias"""
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    bias = (true + pred) / (2 * true)
    return bias.mean()


def MARE_torch(pred, true, mask_value=None):
    """Mean Absolute Relative Error"""
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.div(torch.sum(torch.abs((true - pred))), torch.sum(true))


def SMAPE_torch(pred, true, mask_value=None):
    """Symmetric Mean Absolute Percentage Error"""
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred) / (torch.abs(true) + torch.abs(pred)))


def All_Metrics(pred, true, mask1, mask2):
    """Compute all metrics"""
    assert type(pred) == type(true)
    
    if type(pred) == torch.Tensor:
        mae = MAE_torch(pred, true, mask1)
        rmse = RMSE_torch(pred, true, mask1)
        rrse = RRSE_torch(pred, true, mask1)
    else:
        raise TypeError
    
    return mae, rmse, rrse


def pearson_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient between two PyTorch tensors.
    
    Args:
        x (torch.Tensor): First input tensor.
        y (torch.Tensor): Second input tensor.
    
    Returns:
        torch.Tensor: Pearson correlation coefficient.
    """
    # Ensure the tensors are of type float32
    x = x.float()
    y = y.float()
    
    # Compute the mean of each tensor
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    
    # Compute the deviations from the mean
    dev_x = x - mean_x
    dev_y = y - mean_y
    
    # Compute the covariance between x and y
    covariance = torch.sum(dev_x * dev_y)
    
    # Compute the standard deviations of x and y
    std_x = torch.sqrt(torch.sum(dev_x ** 2))
    std_y = torch.sqrt(torch.sum(dev_y ** 2))
    
    # Compute the Pearson correlation coefficient
    pearson_corr = covariance / (std_x * std_y)
    
    return pearson_corr


def rank_tensor(x):
    """
    Return the ranks of elements in a tensor.
    
    Args:
        x (torch.Tensor): Input tensor.
    
    Returns:
        torch.Tensor: Ranks of the input tensor elements.
    """
    # Get the sorted indices
    sorted_indices = torch.argsort(x)
    
    # Create an empty tensor to hold the ranks
    ranks = torch.zeros_like(sorted_indices, dtype=torch.float)
    
    # Assign ranks based on sorted indices
    ranks[sorted_indices] = torch.arange(1, len(x) + 1).float()
    
    return ranks


def rank_information_coefficient(x, y):
    """
    Calculate the Rank Information Coefficient (RIC) or Spearman's Rank Correlation Coefficient.
    
    Args:
        x (torch.Tensor): First input tensor.
        y (torch.Tensor): Second input tensor.
    
    Returns:
        torch.Tensor: Rank Information Coefficient (RIC).
    """
    # Get the ranks of the elements in x and y
    rank_x = rank_tensor(x)
    rank_y = rank_tensor(y)
    
    # Calculate the mean rank for both tensors
    mean_rank_x = torch.mean(rank_x)
    mean_rank_y = torch.mean(rank_y)
    
    # Calculate the covariance of the rank variables
    covariance = torch.sum((rank_x - mean_rank_x) * (rank_y - mean_rank_y))
    
    # Calculate the standard deviations of the ranks
    std_rank_x = torch.sqrt(torch.sum((rank_x - mean_rank_x) ** 2))
    std_rank_y = torch.sqrt(torch.sum((rank_y - mean_rank_y) ** 2))
    
    # Calculate the Spearman rank correlation (RIC)
    ric = covariance / (std_rank_x * std_rank_y)
    
    return ric
