from data_utils import fetch_constant_info
from utils import *
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import argparse
import torch
import os

SOLVERS = ["dopri8", "dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams', "adaptive_heun", "euler"]
parser = argparse.ArgumentParser('ClimODE')

parser.add_argument('--solver', type=str, default="euler", choices=SOLVERS)
parser.add_argument('--atol', type=float, default=5e-3)
parser.add_argument('--rtol', type=float, default=5e-3)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")
parser.add_argument('--scale', type=int, default=0)
parser.add_argument('--days', type=int, default=3)
parser.add_argument('--spectral', type=int, default=0, choices=[0, 1])
parser.add_argument('--data_path', type=str, default='era_5_data')
parser.add_argument('--vel_folder', type=str, default='velocity_data')
parser.add_argument('--checkpoint', type=str, default='checkpoints/ClimODE_global.pt')
parser.add_argument('--save_path', type=str, default='evaluation/eval.nc')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cwd = os.getcwd()
train_time_scale = slice('2006', '2016')
val_time_scale = slice('2016', '2016')
test_time_scale = slice('2017', '2018')

data_folders = ['geopotential_500/*.nc', 'temperature_850/*.nc', '2m_temperature/*.nc', '10m_u_component_of_wind/*.nc',
                '10m_v_component_of_wind/*.nc']
const_folders = ['constants/constants_5.625deg.nc']
paths_to_data = [os.path.join(args.data_path, folder) for folder in data_folders]
const_info_path = [os.path.join(args.data_path, folder) for folder in const_folders]

print(f"paths_to_data : {paths_to_data}")

levels = ["z", "t", "t2m", "u10", "v10"]
paths_to_data = paths_to_data[0:5]
levels = levels[0:5]
K = len(levels)
assert len(paths_to_data) == len(levels), "Paths to different type of data must be same as number of types of observations"
print("############################ Data is loading ###########################")
Final_train_data = 0
Final_val_data = 0
Final_test_data = 0
max_lev = []
min_lev = []
for idx, data in enumerate(paths_to_data):
    Train_data, Val_data, Test_data, time_steps, lat, lon, mean, std, time_stamp = get_train_test_data_without_scales_batched(
        data, train_time_scale, val_time_scale, test_time_scale, levels[idx]
        )
    max_lev.append(mean)
    min_lev.append(std)
    if idx == 0:
        Final_train_data = Train_data
        Final_val_data = Val_data
        Final_test_data = Test_data
    else:
        Final_train_data = torch.cat([Final_train_data, Train_data], dim=2)
        Final_val_data = torch.cat([Final_val_data, Val_data], dim=2)
        Final_test_data = torch.cat([Final_test_data, Test_data], dim=2)

print("Length of training data", len(Final_train_data))
print("Length of validation data", len(Final_val_data))
print("Length of testing data", len(Final_test_data))
const_channels_info, lat_map, lon_map = fetch_constant_info(const_info_path)

if args.spectral == 1: print("############## Running the Model in Spectral Domain ####################")
H, W = Train_data.shape[3], Train_data.shape[4]
clim = torch.mean(Final_test_data, dim=0)
Test_loader = DataLoader(Final_test_data[2:], batch_size=args.batch_size, shuffle=False)
time_loader = DataLoader(time_steps[2:], batch_size=args.batch_size, shuffle=False)
time_idx_steps = torch.tensor([i for i in range(365 * 4)]).view(-1, 1)
time_idx = DataLoader(time_idx_steps[2:], batch_size=args.batch_size, shuffle=False, pin_memory=False)

total_time_len = len(time_steps[2:])
total_time_steps = time_steps[2:].numpy().flatten().tolist()
num_years = 2
Final_train_data = 0
Final_val_data = 0

vel_test_path = os.path.join(args.vel_folder, 'test_10year_2day_mm_vel.npy')
vel_test = torch.from_numpy(np.load(vel_test_path))

model = torch.load(args.checkpoint, map_location=torch.device('cpu')).to(device)
model.set_constants(const_channels_info.to(device), lat_map.to(device), lon_map.to(device))
print(model)

RMSD = []
RMSD_lat_lon = []
Pred = []
Truth = []

Lead_RMSD_arr = {level_:np.zeros(args.batch_size-1) for level_ in levels}
Lead_ACC = {level_:np.zeros(args.batch_size-1) for level_ in levels}
Lead_CRPS = {level_:np.zeros((args.batch_size-1, H, W)) for level_ in levels}

predicted_states = np.zeros((num_years, K, H, W, total_time_len))

for entry, (time_steps, batch) in enumerate(zip(time_loader, Test_loader)):
    data = batch[0].to(device).view(num_years, 1, len(paths_to_data) * (args.scale + 1), H, W)
    past_sample = vel_test[entry].view(num_years, 2 * len(paths_to_data) * (args.scale + 1), H, W).to(device)
    t = time_steps.float().to(device).flatten()
    mean_pred, std_pred, mean_wo_bias = model(data, past_sample, t)
    mean_avg = mean_pred.view(-1, len(paths_to_data) * (args.scale + 1), H, W)
    std_avg = std_pred.view(-1, len(paths_to_data) * (args.scale + 1), H, W)

    timesteps_start_idx = entry*args.batch_size
    n_timesteps = len(t)
    timesteps_end_idx = timesteps_start_idx + n_timesteps
    # TODO : verify order of this and change to put time before K
    predicted_states[:, :, :, :,timesteps_start_idx:timesteps_end_idx] = mean_pred.detach().cpu().numpy().reshape(num_years, K, H, W, n_timesteps)

    for yr in range(2):
        for t_step in range(1, len(time_steps)):
            evaluate_rmsd = evaluation_rmsd_mm(
                mean_pred[t_step, yr, :, :, :].cpu(), batch[t_step, yr, :, :, :].cpu(), lat, lon, max_lev, min_lev, H, W, levels
            )
            evaluate_acc = anomaly_correlation_coefficient(
                mean_pred[t_step, yr, :, :, :].cpu(), batch[t_step, yr, :, :, :].cpu(), lat, lon, max_lev, min_lev, H, W, levels,
                clim[yr, :, :, :].cpu()
            )
            evaluate_crps = evaluation_crps_mm(
                mean_pred[t_step, yr, :, :, :].cpu(), batch[t_step, yr, :, :, :].cpu(), lat, lon, max_lev, min_lev, H, W, levels,
                std_pred[t_step, yr, :, :, :].cpu()
            )
            for idx, lev in enumerate(levels):
                Lead_RMSD_arr[lev][t_step-1] = evaluate_rmsd[idx]
                Lead_ACC[lev][t_step-1] = evaluate_acc[idx]
                Lead_CRPS[lev][t_step-1] = evaluate_crps[idx]


predicted_states = predicted_states.reshape(K,H,W,num_years*total_time_len)

# Save predictions for further display
lat = np.linspace(-90, 90, H)    # H points from -90 to 90 (latitude)
lon = np.linspace(-180, 180, W)  # W points from -180 to 180 (longitude)
time = np.arange(num_years*total_time_len)
variables = {level_:(['time', 'lat', 'lon'], predicted_states[i]) for i, level_ in enumerate(levels)}
predicted_dataset = xr.Dataset(data_vars=variables, coords={'time': time, 'lat': lat, 'lon': lon})


if not os.path.exists(os.path.dirname(args.save_path)):
    os.makedirs(os.path.dirname(args.save_path))
predicted_dataset.to_netcdf(args.save_path)


for t_idx in range(args.batch_size-1):
    for idx, lev in enumerate(levels):
        print(
            "Lead Time ", (t_idx + 1) * 6, "hours ", "| Observable ", lev, "| Mean RMSD ", np.mean(Lead_RMSD_arr[lev][t_idx]), "| Std RMSD ",
            np.std(Lead_RMSD_arr[lev][t_idx])
        )
        print(
            "Lead Time ", (t_idx + 1) * 6, "hours ", "| Observable ", lev, "| Mean ACC ", np.mean(Lead_ACC[lev][t_idx]), "| Std ACC ",
            np.std(Lead_ACC[lev][t_idx])
        )
        print(
            "Lead Time ", (t_idx + 1) * 6, "hours ", "| Observable ", lev, "| Mean CRPS ", np.mean(Lead_CRPS[lev][t_idx]), "| Std CRPS ",
            np.std(Lead_CRPS[lev][t_idx])
        )
