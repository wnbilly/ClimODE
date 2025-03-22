import numpy as np
import random
import os
import torch

BOUNDARIES = {
    'NorthAmerica': {  # 8x14
        'lat_range': (15, 65),
        'lon_range': (220, 300)
    },
    'SouthAmerica': {  # 14x10
        'lat_range': (-55, 20),
        'lon_range': (270, 330)
    },
    'Europe'      : {  # 6x8
        'lat_range': (30, 65),
        'lon_range': (0, 40)
    },
    'SouthAsia'   : {  # 10, 14
        'lat_range': (-15, 45),
        'lon_range': (25, 110)
    },
    'EastAsia'    : {  # 10, 12
        'lat_range': (5, 65),
        'lon_range': (70, 150)
    },
    'Australia'   : {  # 10x14
        'lat_range': (-50, 10),
        'lon_range': (100, 180)
    }
}


def save_checkpoint(model, optimizer, scheduler, epoch, file_path, verbose=False):
    checkpoint = {
        'epoch'    : epoch,
        'model'    : model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(checkpoint, file_path)
    if verbose:
        print(f"Checkpoint saved to {file_path}")


def load_checkpoint(file_path, model, optimizer=None, scheduler=None, device=None):
    # Note: Input model & optimizer should be pre-defined. This routine only updates their states.
    start_epoch = 0
    if os.path.exists(file_path):
        print(f"Loading checkpoint {file_path}")
        checkpoint = torch.load(file_path, weights_only=False, map_location=device)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"Loaded checkpoint at epoch {start_epoch}")
    else:
        print(f"No checkpoint found at {file_path}")

    return model, optimizer, scheduler, start_epoch


def verif_path(path_):
    if not os.path.exists(path_):
        os.makedirs(path_)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")


# Section 3.8 : Loss
def negative_log_likelihood(mean, std, truth, var_coeff):
    predicted_gaussian_distribution = torch.distributions.normal.Normal(mean, std + 1e-3)
    observations_negative_log_likelihood = - predicted_gaussian_distribution.log_prob(truth)
    # Equation (12) with var_coeff being the hypervariance (variance weight) parameter
    loss = observations_negative_log_likelihood.mean() + var_coeff * (std ** 2).sum()
    return loss


def custom_loss(mean, std, truth, var_coeff):
    return torch.nn.functional.mse_loss(mean, truth, reduction='mean') + var_coeff * (std ** 2).sum()


def latitude_weighted_rmse(target, predicted, latitude_grid, latitude_is_rad=False):
    '''

    :param target: (batch_size, n_timesteps, n_levels, lat, lon)
    :param predicted: (batch_size, n_timesteps, n_levels, lat, lon)
    :param latitude_grid: latitude in radians
    :return:
    '''

    batch_dim = 0
    temporal_dim = 1
    spatial_dims = (3, 4)

    if not latitude_is_rad:
        latitude_grid = torch.deg2rad(latitude_grid)

    latitude_weights = torch.cos(latitude_grid)
    latitude_weights /= latitude_weights.mean()

    spatial_rmse = torch.mean(latitude_weights * ((predicted - target) ** 2), dim=spatial_dims)

    lat_rmse_per_sample_per_level = torch.mean(torch.sqrt(spatial_rmse), dim=temporal_dim)

    return torch.sum(lat_rmse_per_sample_per_level, dim=batch_dim)


def anomaly_correlation_coefficient(target, predicted, latitude_grid, latitude_is_rad=False, reduction='sum'):
    '''

    :param target: (batch_size, n_timesteps, n_levels, lat, lon)
    :param predicted: (batch_size, n_timesteps, n_levels, lat, lon)
    :param latitude_grid:
    :param latitude_is_rad:
    :param reduction:
    :return:
    '''
    target_temporal_mean = target.mean(dim=1, keepdim=True)  #  time dim is 1

    batch_dim = 0
    spatio_temporal_dims = (1, 3, 4)

    if not latitude_is_rad:
        latitude_grid = torch.deg2rad(latitude_grid)

    latitude_weights = torch.cos(latitude_grid)
    latitude_weights /= latitude_weights.mean()

    target_tilde = target - target_temporal_mean
    predicted_tilde = predicted - target_temporal_mean

    acc_per_sample_per_level = (torch.sum(latitude_weights * target_tilde * predicted_tilde, dim=spatio_temporal_dims) / torch.sqrt(
        torch.sum(latitude_weights * (target_tilde ** 2), dim=spatio_temporal_dims) * torch.sum(
            latitude_weights * (predicted_tilde ** 2), dim=spatio_temporal_dims
        )
    ))

    return torch.sum(acc_per_sample_per_level, dim=batch_dim)
