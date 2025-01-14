import numpy as np
import xarray as xr
import random
import os
import torch
import properscoring as ps


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


def save_checkpoint(model, optimizer, scheduler, epoch, file_path='checkpoint.pt'):
    checkpoint = {
        'epoch'    : epoch,
        'model'    : model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved to {file_path}")


def load_checkpoint(model, optimizer, scheduler, file_path='checkpoint.pt'):
    # Note: Input model & optimizer should be pre-defined. This routine only updates their states.
    start_epoch = 0
    if os.path.exists(file_path):
        print(f"Loading checkpoint {file_path}")
        checkpoint = torch.load(file_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"Loaded checkpoint at epoch {start_epoch}")
    else:
        print(f"No checkpoint found at {file_path}")

    return model, optimizer, scheduler, start_epoch


def verif_path(path_):
    if os.path.isdir(path_):
        os.makedirs(path_)
    else:
        dirname_ = os.path.dirname(path_)
        if not os.path.exists(dirname_):
            os.makedirs(dirname_)

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
    normal_lkl = torch.distributions.normal.Normal(mean, 1e-3 + std)
    observations_negative_log_likelihood = - normal_lkl.log_prob(truth)
    # Equation (12) with var_coeff being the hypervariance (variance weight) parameter
    loss = observations_negative_log_likelihood.mean() + var_coeff * (std ** 2).sum()
    return loss


def evaluation_rmsd_mm(pred, truth, latitude, longitude, max_vals, min_vals, H, W, levels):
    RMSD_final = []
    RMSD_lat_lon = []
    true_lat_lon = []
    pred_lat_lon = []
    for idx, lev in enumerate(levels):
        true_idx = idx
        das_pred = []
        das_true = []
        pred_spectral = pred[idx].detach().cpu().numpy()
        true_spectral = truth[true_idx, :, :].detach().cpu().numpy()

        curr_pred = pred_spectral * (max_vals[idx] - min_vals[idx]) + min_vals[idx]

        das_pred.append(
            xr.DataArray(curr_pred.reshape(1, H, W), dims=['time', 'lat', 'lon'], coords={'time': [0], 'lat': latitude, 'lon': longitude}, name=lev)
        )
        pred_xr = xr.merge(das_pred)

        true = true_spectral * (max_vals[idx] - min_vals[idx]) + min_vals[idx]

        das_true.append(
            xr.DataArray(true.reshape(1, H, W), dims=['time', 'lat', 'lon'], coords={'time': [0], 'lat': latitude, 'lon': longitude}, name=lev)
        )
        True_xr = xr.merge(das_true)
        error = pred_xr - True_xr
        weights_lat = np.cos(np.deg2rad(error.lat))
        weights_lat /= weights_lat.mean()
        rmse = np.sqrt(((error) ** 2 * weights_lat).mean(dim=['lat', 'lon'])).mean(dim=['time'])
        lat_lon_rmse = np.sqrt((error) ** 2)
        RMSD_lat_lon.append(lat_lon_rmse[lev].values)
        RMSD_final.append(rmse[lev].values.tolist())

    return RMSD_final


def anomaly_correlation_coefficient(pred, truth, lat, lon, max_vals, min_vals, H, W, levels, clim):
    acc_list = []

    for idx, lev in enumerate(levels):
        pred_spectral = pred[idx].detach().cpu().numpy()
        true_spectral = truth[idx, :, :].detach().cpu().numpy()
        pred_spectral = pred_spectral - clim[idx].detach().numpy()
        true_spectral = true_spectral - clim[idx].detach().numpy()

        curr_pred = pred_spectral * (max_vals[idx] - min_vals[idx]) + min_vals[idx]
        true = true_spectral * (max_vals[idx] - min_vals[idx]) + min_vals[idx]

        weights_lat = np.cos(np.deg2rad(lat))
        weights_lat /= weights_lat.mean()
        weights_lat = weights_lat.reshape(len(lat), 1)
        weights_lat = weights_lat.repeat(len(lon), 1)

        pred_prime = curr_pred - np.mean(curr_pred)
        true_prime = true - np.mean(true)

        acc = np.sum(weights_lat * pred_prime * true_prime) / np.sqrt(np.sum(weights_lat * pred_prime ** 2) * np.sum(weights_lat * true_prime ** 2))
        acc_list.append(acc)

    return acc_list


def evaluation_crps_mm(pred, truth, lat, lon, max_vals, min_vals, H, W, levels, sigma):
    CRPS_final = []

    for idx, lev in enumerate(levels):
        pred_spectral = pred[idx].detach().cpu().numpy()
        true_spectral = truth[idx, :, :].detach().cpu().numpy()
        std_spectral = sigma[idx].detach().cpu().numpy()

        curr_pred = pred_spectral * (max_vals[idx] - min_vals[idx]) + min_vals[idx]
        true = true_spectral * (max_vals[idx] - min_vals[idx]) + min_vals[idx]

        crps = ps.crps_gaussian(true_spectral, mu=pred_spectral, sig=std_spectral)
        CRPS_final.append(crps)

    return CRPS_final
