from torch.utils.data import DataLoader
import numpy as np
import argparse
import torch.optim as optim
import torch
import os
from tqdm import tqdm

# local imports
from models import ClimateEncoderFreeUncertain
from utils import set_seed, negative_log_likelihood, load_checkpoint, save_checkpoint, verif_path, anomaly_correlation_coefficient, \
    latitude_weighted_rmse
from data_utils import fetch_constant_info, get_resampled_normalized_data, load_velocity, utvDataset, save_pickle, rescale_minmax, load_pickle

SOLVERS = ["dopri8", "dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams', "adaptive_heun", "euler"]
parser = argparse.ArgumentParser('ClimODE evaluation')
parser.add_argument('--solver', type=str, default="euler", choices=SOLVERS)
parser.add_argument('--atol', type=float, default=5e-3)
parser.add_argument('--rtol', type=float, default=5e-3)
parser.add_argument('--delta_t', type=int, default=6, help="Time step between data (in hours)")
parser.add_argument('--n_epochs', type=int, default=300, help='Number of epochs to train for.')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_steps', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--l2_lambda', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--data_path', type=str, default='era_5_data')
parser.add_argument('--model_save_folder', type=str, default='checkpoints')
parser.add_argument('--checkpoint_path', type=str, default='checkpoints/checkpoint.pt')
parser.add_argument('--velocity_data_path', type=str)
parser.add_argument('--eval_save_path', type=str, default=None)
parser.add_argument('--device', type=str, default='auto')
args = parser.parse_args()

#  args processing
verif_path(args.model_save_folder)
verif_path(os.path.dirname(args.checkpoint_path))

data_scales_path = 'train_data_scales.pkl'

if args.eval_save_path is None:
    default_save_path = 'evaluation'
    verif_path(default_save_path)
    args.eval_save_path = os.path.join(default_save_path, os.path.basename(args.velocity_data_path).replace('nc', 'pkl'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
print(f"device : {device}")

levels = ["z", "t", "t2m", "u10", "v10"]
data_folders = ['geopotential_500/*.nc', 'temperature_850/*.nc', '2m_temperature/*.nc', '10m_u_component_of_wind/*.nc',
                '10m_v_component_of_wind/*.nc']
constants = ['orography', 'lsm']
const_folders = ['constants/constants_5.625deg.nc']
paths_to_data = [os.path.join(args.data_path, folder) for folder in data_folders]
const_info_path = [os.path.join(args.data_path, folder) for folder in const_folders]

#  The timestamps of the given velocity file are the reference timestamps
velocity, timestamps = load_velocity(args.velocity_data_path)
min_date, max_date = timestamps[0], timestamps[-1]

data_scales = load_pickle(data_scales_path)
data = torch.cat(
    [get_resampled_normalized_data(
        level_path, str(min_date), str(max_date), levels[idx], scales=[data_scales['min'][idx].item(), data_scales['max'][idx].item()]
        )[0] for idx, level_path in enumerate(paths_to_data)], dim=1
    )

constants_to_fetch = ['orography', 'lsm']
# const_channels_info : tensor of shape (1, n_constants, H, W)
const_channels_info, lat_map, lon_map = fetch_constant_info(const_info_path, constants_to_fetch)

n_constants = len(constants_to_fetch)
n_quantities = len(paths_to_data)  # K in the paper
H, W = data.shape[2], data.shape[3]  #  H : number of points latitude-wise, W : number of points longitude-wise

dataset = utvDataset(data, velocity, timestamps, n_steps=args.n_steps, window_step=args.n_steps)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.custom_collate_fn)

# Model declaration
model = ClimateEncoderFreeUncertain(
    n_quantities, n_constants, out_channels=n_quantities, ode_solver=args.solver, use_attention=True, use_uq=True, gamma=0.1
    ).to(
    device
)
model.set_constants(const_channels_info.to(device), lat_map.to(device), lon_map.to(device))

if os.path.exists(args.checkpoint_path):
    #  Load model
    model, _, _, _ = load_checkpoint(args.checkpoint_path, model, device=device)

model.eval()
mse = torch.nn.MSELoss(reduction='sum')

# predicted = {}

total_mse = .0
total_lat_rmse = torch.zeros((n_quantities), device=device)
total_acc = torch.zeros((n_quantities), device=device)

with torch.no_grad():
    for batch_idx, batch in enumerate(dataloader):
        u = batch['u'].to(device)
        u0 = u[:, 0].to(device)
        t = batch['t'].to(device)
        v0 = batch['v0'].to(device)

        current_batch_size = len(u)

        u_pred_w_bias, uncertainty, u_pred = model(u0, v0, t, args.delta_t)
        loss = mse(u_pred_w_bias, u).item()

        u = rescale_minmax(u, data_scales['min'], data_scales['max'])
        u_pred_w_bias = rescale_minmax(u_pred_w_bias, data_scales['min'], data_scales['max'])
        u_pred = rescale_minmax(u_pred, data_scales['min'], data_scales['max'])

        lat_rmse = latitude_weighted_rmse(u_pred_w_bias, u, lat_map.to(device))
        acc = anomaly_correlation_coefficient(u_pred_w_bias, u, lat_map.to(device))

        '''predicted.update(
            {str(batch['timestamps'][idx][0]): {
                'timestamps':batch['timestamps'][idx],
                'u_gt': u[idx].numpy(force=True), 'u_pred_w_bias': u_pred_w_bias[idx].numpy(force=True),'u_pred_no_bias': u_pred[idx].numpy(force=True),
                'uncertainty': uncertainty[idx].numpy(force=True)
            } for idx in range(current_batch_size)}
            )'''

        print(
            f"Batch {batch_idx}\n"
            f"MSE : {loss / current_batch_size}\n"
            f"Latitude-weighted RMSE : {lat_rmse / current_batch_size}\n"
            f"ACC : {acc / current_batch_size}\n"
            f"{'-' * 20}"
            )

        total_mse += loss
        total_lat_rmse += lat_rmse
        total_acc += acc

mse = total_mse / len(dataset)
acc = total_acc / len(dataset)  #  Average
acc_dict = {level: acc[level_idx].item() for level_idx, level in enumerate(levels)}  #  Link to level names
lat_rmse = total_lat_rmse / len(dataset)
lat_rmse_dict = {level: lat_rmse[level_idx].item() for level_idx, level in enumerate(levels)}

print(
    f"Final results for {os.path.basename(args.velocity_data_path)} :\n"
    f"{len(dataset)} samples\n"
    f"MSE : {mse}\n"
    f"Latitude-weighted RMSE : {lat_rmse_dict}\n"
    f"ACC : {acc_dict}"
)

# save_pickle(predicted, args.eval_save_path)
