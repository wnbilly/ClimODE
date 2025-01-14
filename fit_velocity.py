import xarray as xr
from datetime import datetime, timedelta
import argparse
import torch
import os
import numpy as np

# local imports
from utils import set_seed
from data_utils import fetch_constant_info, get_resampled_normalized_data
from velocity_utils import get_gauss_kernel, fit_velocity, OptimVelocity

torch.cuda.empty_cache()
set_seed(42)

parser = argparse.ArgumentParser('ClimODE velocity fitter')

parser.add_argument('--time_sampling', type=int, default=6, help="Time step between data (in hours)")
parser.add_argument('--n_samples', type=int, default=3, help="The number of samples necessary to compute the derivatives")
parser.add_argument('--sequence_length', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--data_path', type=str, default='era_5_data')
parser.add_argument('--save_path', type=str)
parser.add_argument(
    '--min_date', type=datetime.fromisoformat, default=datetime(2016, 1, 1),
    help='min date to compute velocity. It should take into account that previous samples are required to compute velocity'
)
parser.add_argument('--max_date', type=datetime.fromisoformat, default=datetime(2016, 12, 31), help='max date to compute velocity')
parser.add_argument('--kernel_path', type=str, default='kernel.npy')
args = parser.parse_args()

#  args processing
if args.save_path is None:
    date_format = '%Y-%m-%d'
    args.save_path = os.path.join(
        'velocity_data', f"fitted_velocity_{args.min_date.strftime(date_format)}_to_{args.max_date.strftime(date_format)}.nc"
        )

if not os.path.exists(os.path.dirname(args.save_path)):
    os.makedirs(os.path.dirname(args.save_path))

# Add the necessary timestamps to compute velocity
args.min_date = args.min_date - timedelta(hours=args.time_sampling * args.n_samples, minutes=-1)

levels = ["z", "t", "t2m", "u10", "v10"]
data_folders = ['geopotential_500/*.nc', 'temperature_850/*.nc', '2m_temperature/*.nc', '10m_u_component_of_wind/*.nc',
                '10m_v_component_of_wind/*.nc']
constants = ['orography', 'lsm']
const_folders = ['constants/constants_5.625deg.nc']
paths_to_data = [os.path.join(args.data_path, folder) for folder in data_folders]
const_info_path = [os.path.join(args.data_path, folder) for folder in const_folders]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device : {device}")

data = 0

for idx, level_path in enumerate(paths_to_data):
    # loads level by level
    norm_data, timestamps = get_resampled_normalized_data(level_path, args.min_date, args.max_date, levels[idx])

    if idx == 0:
        data = norm_data
    else:
        data = torch.cat([data, norm_data], dim=1)

# const_channels_info : tensor of shape (1, n_constants, H, W)
const_channels_info, lat_map, lon_map = fetch_constant_info(const_info_path, constants)

# velocity fitting
H, W = data.shape[2], data.shape[3]  #  H : number of points latitude-wise, W : number of points longitude-wise
kernel = torch.from_numpy(np.load(args.kernel_path)) if os.path.exists(args.kernel_path) else get_gauss_kernel((H, W), lat_map[:, 0], lon_map[0])

velocity = fit_velocity(data, args.time_sampling, vel_model=OptimVelocity, kernel=kernel, device=device, n_samples=args.n_samples)
velocity = velocity.detach().numpy()

time = timestamps[-len(velocity):]  #  keep the timestamps for which the velocity could be computed
lat, lon = lat_map[:, 0], lon_map[0]
variables = {f"{level_}_vel": (['time', 'deriv_dim', 'lat', 'lon'], velocity[:, :, i]) for i, level_ in enumerate(levels)}

velocity_dataset = xr.Dataset(data_vars=variables, coords={'time': time, 'deriv_dim': np.arange(2), 'lat': lat, 'lon': lon})
velocity_dataset.to_netcdf(args.save_path)
