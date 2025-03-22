import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl

from data_utils import load_pickle
from utils import verif_path

# Enable anti-aliasing globally
mpl.rcParams['text.antialiased'] = True
mpl.rcParams['lines.antialiased'] = True
mpl.use('Agg')

PROJECTION = ccrs.PlateCarree()
FIGSIZE = (6, 5)
DPI = 200
FPS = 2
BITRATE = 3000

ABSOLUTE_ZERO = -273.15  # in degrees celsius
SEED = 0


def kelvin_to_celsius(temp):
    return temp + ABSOLUTE_ZERO


def roll_longitude(array, shift):
    if isinstance(array, torch.Tensor):
        return torch.roll(array, shift, dims=2)
    elif isinstance(array, np.ndarray):
        return np.roll(array, shift, axis=2)
    else:
        print(f"Could not roll {array.__class__.__name__}")
        return array


def compute_animation_for_vectors(vectors, save_path, titles=None, cbar_label=None, lat_ext=[-90, 90], lon_ext=[-180, 180], suptitle=None, roll=32):
    """

    :param vectors: (n_scalars, time, dimensions(lat and lon), lat, lon)
    :param lat_ext:
    :param lon_ext:
    :param save_path:
    :return:
    """
    n_subplots = len(vectors)
    n_timesteps = vectors[0].shape[0]
    colormap = 'viridis'

    H, W = vectors[0].shape[-2:]
    lat = np.linspace(lat_ext[0], lat_ext[1], H)  # H points from -90 to 90 (latitude)
    lon = np.linspace(lon_ext[0], lon_ext[1], W)  # W points from -180 to 180 (longitude)
    lon2d, lat2d = np.meshgrid(lon, lat)

    if roll != 0:
        for vector_idx, vector in enumerate(vectors):
            vectors[vector_idx] = roll_longitude(vector, roll)

    # Set up the figure and axis
    fig, axs = plt.subplots(nrows=n_subplots, subplot_kw={'projection': PROJECTION}, figsize=FIGSIZE, dpi=DPI)

    if suptitle is not None:
        fig.suptitle(suptitle)

    if n_subplots == 1:
        axs = [axs]

    # Compute the wind magnitude
    magnitudes = [np.sqrt(vectors[i][:, 0] ** 2 + vectors[i][:, 1] ** 2) for i in range(n_subplots)]

    # To have a consistent colorbar over all the busplots
    vmin = min([s_.min() for s_ in magnitudes])
    vmax = min([s_.max() for s_ in magnitudes])
    normalizer = plt.Normalize(vmin, vmax)
    cbar_mappable = plt.cm.ScalarMappable(norm=normalizer, cmap=colormap)
    cbar = fig.colorbar(cbar_mappable, ax=axs)
    cbar.set_label(cbar_label)

    def update_vectors(frame):

        for i, ax in enumerate(axs):
            ax.clear()
            ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
            ax.coastlines()
            ax.set_title(titles[i])  # Customize title as needed

            # Plot the wind vectors with color based on magnitude
            ax.quiver(
                lon2d, lat2d,
                vectors[i][frame, 0], vectors[i][frame, 1],
                magnitudes[i][frame],
                transform=ccrs.PlateCarree(),
                # scale=1,
                scale_units='xy',
                width=0.0018,
                cmap=colormap
            )

    # Create an animation
    ani = FuncAnimation(fig, update_vectors, frames=n_timesteps, interval=200, blit=False)

    # Save the animation with high quality
    ani.save(save_path, writer="ffmpeg", fps=FPS, dpi=DPI, bitrate=BITRATE)
    plt.close()


def compute_animation_for_scalars(scalars, save_path, titles=None, cbar_label=None, lat_ext=[-90, 90], lon_ext=[-180, 180], suptitle=None, roll=32):
    """

    :param scalars: (n_scalars, time, lat, lon)
    :param lat_ext:
    :param lon_ext:
    :param save_path:
    :return:
    """
    n_scalars = len(scalars)
    n_timesteps = scalars[0].shape[0]
    colormap = 'coolwarm'

    extent = lon_ext + lat_ext

    if roll != 0:
        for scalar_idx, scalar in enumerate(scalars):
            scalars[scalar_idx] = roll_longitude(scalar, roll)

    # Set up the figure and axis
    fig, axs = plt.subplots(nrows=n_scalars, subplot_kw={'projection': PROJECTION}, figsize=FIGSIZE, dpi=DPI)

    if suptitle is not None:
        fig.suptitle(suptitle)

    if n_scalars == 1:
        axs = [axs]

    for i, ax in enumerate(axs):
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
        ax.coastlines()
        ax.set_title(titles[i])  # Customize title as needed

    # To have a consistent colorbar over all the busplots
    vmin = min([s_.min() for s_ in scalars])
    vmax = min([s_.max() for s_ in scalars])
    normalizer = plt.Normalize(vmin, vmax)
    cbar_mappable = plt.cm.ScalarMappable(norm=normalizer, cmap=colormap)
    cbar = fig.colorbar(cbar_mappable, ax=axs)
    cbar.set_label(cbar_label)

    def update_scalars(frame):
        # ax.clear()
        for i, ax in enumerate(axs):
            ax.imshow(scalars[i][frame], origin='lower', extent=extent, cmap=colormap, norm=normalizer)

    # Create an animation
    ani = FuncAnimation(fig, update_scalars, frames=n_timesteps, interval=200, blit=False)

    # Save the animation with high quality
    ani.save(save_path, writer="ffmpeg", fps=FPS, dpi=DPI, bitrate=BITRATE)
    plt.close()


if __name__ == '__main__':
    # Load the NetCDF file
    import argparse
    import os

    levels = ["z", "t", "t2m", "u10", "v10"]
    levels_idx = {level_: i for i, level_ in enumerate(levels)}
    levels_nature = {'temperature': ['t', 't2m'], 'wind': ['u10', 'v10'], 'other': ['z']}

    parser = argparse.ArgumentParser('ClimODE states display')
    parser.add_argument('--data_path', type=str, default='era_5_data')
    parser.add_argument('--levels', nargs='*', type=str, default=levels)
    parser.add_argument('--n_points', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='evaluation/animations')
    parser.add_argument('--extra', type=str, default='')
    args = parser.parse_args()

    file_path = args.data_path  # Replace with your NetCDF file path
    data = load_pickle(file_path)
    verif_path(args.save_path)

    temp_levels = list(filter(lambda x: x in args.levels, levels_nature['temperature']))
    DO_WIND = 'u10' in args.levels and 'v10' in args.levels
    other_levels = list(filter(lambda x: x in args.levels, levels_nature['other']))

    if args.n_points is not None:
        args.n_points = min(args.n_points, len(data))
        timestamps = list(data.keys())
        rng = np.random.default_rng(seed=SEED)
        chosen_timestamps = rng.choice(timestamps, args.n_points, replace=False)
        data = {ts_: data[ts_] for ts_ in chosen_timestamps}

    for ts in data:
        #  Temperature
        for level in temp_levels:
            idx = levels_idx[level]
            gt = kelvin_to_celsius(data[ts]['u_gt'][:, idx])
            pred = kelvin_to_celsius(data[ts]['u_pred_w_bias'][:, idx])
            save_path = os.path.join(args.save_path, str(level))
            verif_path(save_path)
            save_path = os.path.join(save_path, f'{ts}{args.extra}.mp4')
            compute_animation_for_scalars(
                [gt, pred], save_path, titles=[f'{level}_gt', f'{level}_pred'], cbar_label='Temperature °C'
            )

        #  Wind
        if DO_WIND:
            #  TODO : see https://confluence.ecmwf.int/display/CKB/Copernicus+Arctic+Regional+Reanalysis+%28CARRA%29%3A+Data+User+Guide#CopernicusArcticRegionalReanalysis(CARRA):DataUserGuide-Variablesat10-metreheight
            wind_gt = np.stack((data[ts]['u_gt'][:, levels_idx['v10']], data[ts]['u_gt'][:, levels_idx['u10']]), axis=1)
            wind_pred = np.stack((data[ts]['u_pred_w_bias'][:, levels_idx['v10']], data[ts]['u_pred_w_bias'][:, levels_idx['u10']]), axis=1)

            save_path = os.path.join(args.save_path, 'wind')
            verif_path(save_path)
            save_path = os.path.join(save_path, f'{ts}{args.extra}.mp4')
            compute_animation_for_vectors(
                [wind_gt, wind_pred], save_path, titles=[f'wind_gt', f'wind_pred'], cbar_label='Wind speed m/s'
            )

        #  Rest
        for level in other_levels:
            idx = levels_idx[level]
            gt = data[ts]['u_gt'][:, idx]
            # compute_animation_for_scalar(t2m_gt, lat, lon, f'evaluation/animations/t2m_gt_animation_{ts}.mp4')
            pred = data[ts]['u_pred_w_bias'][:, idx]
            # compute_animation_for_scalar(t2m_pred, lat, lon, f'evaluation/animations/t2m_pred_w_bias_animation_{ts}.mp4')
            save_path = os.path.join(args.save_path, str(level))
            verif_path(save_path)
            save_path = os.path.join(save_path, f'{ts}{args.extra}.mp4')
            compute_animation_for_scalars([gt, pred], save_path, titles=[f'{level}_gt', f'{level}_pred'])
