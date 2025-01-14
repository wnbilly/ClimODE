from torch.utils.data import DataLoader
import numpy as np
import argparse
import torch.optim as optim
import torch
import os


# local imports
from models import ClimateEncoderFreeUncertain
from utils import set_seed, negative_log_likelihood, load_checkpoint, save_checkpoint, verif_path
from data_utils import fetch_constant_info, get_resampled_normalized_data, load_velocity, utvDataset

torch.cuda.empty_cache()
set_seed(42)

SOLVERS = ["dopri8", "dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams', "adaptive_heun", "euler"]
parser = argparse.ArgumentParser('ClimODE')
parser.add_argument('--solver', type=str, default="euler", choices=SOLVERS)
parser.add_argument('--atol', type=float, default=5e-3)
parser.add_argument('--rtol', type=float, default=5e-3)
parser.add_argument('--delta_t', type=int, default=6, help="Time step between data (in hours)")
parser.add_argument('--n_epochs', type=int, default=300, help='Number of epochs to train for.')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--l2_lambda', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--data_path', type=str, default='era_5_data')
parser.add_argument('--model_save_folder', type=str, default='checkpoints')
parser.add_argument('--checkpoint_path', type=str, default='checkpoints/checkpoint.pt')
parser.add_argument('--velocity_data_path', type=str)
args = parser.parse_args()

#  args processing
verif_path(args.model_save_folder)
verif_path(args.checkpoint_path)

levels = ["z", "t", "t2m", "u10", "v10"]
data_folders = ['geopotential_500/*.nc', 'temperature_850/*.nc', '2m_temperature/*.nc', '10m_u_component_of_wind/*.nc',
                '10m_v_component_of_wind/*.nc']
constants = ['orography', 'lsm']
const_folders = ['constants/constants_5.625deg.nc']
paths_to_data = [os.path.join(args.data_path, folder) for folder in data_folders]
const_info_path = [os.path.join(args.data_path, folder) for folder in const_folders]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device : {device}")

# The timestamps of the given velocity file are the reference timestamps
velocity, timestamps = load_velocity(args.velocity_data_path)
min_date, max_date = timestamps[0], timestamps[-1]

data = torch.cat(
    [get_resampled_normalized_data(level_path, str(min_date), str(max_date), levels[idx])[0] for idx, level_path in enumerate(paths_to_data)], dim=1
)

constants_to_fetch = ['orography', 'lsm']
# const_channels_info : tensor of shape (1, n_constants, H, W)
const_channels_info, lat_map, lon_map = fetch_constant_info(const_info_path, constants_to_fetch)

n_constants = len(constants_to_fetch)
n_quantities = len(paths_to_data)  # K in the paper
H, W = data.shape[2], data.shape[3]  #  H : number of points latitude-wise, W : number of points longitude-wise

dataset = utvDataset(torch.tensor(data), torch.tensor(velocity), timestamps, n_steps=8)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.custom_collate_fn, pin_memory_device=device.type)

# Model declaration
model = ClimateEncoderFreeUncertain(n_quantities, n_constants, out_channels=n_quantities, ode_solver=args.solver, use_attention=True, use_uq=True).to(
    device
)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300)
model.set_constants(const_channels_info.to(device), lat_map.to(device), lon_map.to(device))
start_epoch = 0

if os.path.exists(args.checkpoint_path):
    #  Resume training
    model, optimizer, scheduler, start_epoch, batch_idx = load_checkpoint(model, optimizer, scheduler, args.checkpoint_path)

for epoch in range(start_epoch, args.n_epochs):

    epoch_train_loss = .0
    epoch_val_loss = .0

    # RMSD = []

    if epoch == 0:
        var_coeff = 0.001
    else:
        var_coeff = 2 * scheduler.get_last_lr()[0]

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        u = batch['u']
        u0 = u[:, 0]
        t = batch['t']
        v0 = batch['v0']

        u_pred_w_bias, uncertainty, u_pred = model(u0, v0, t, args.delta_t)

        # Compare computed u(t) (with mean and std) to truth u(t) of batch
        # TODO : revise the loss function
        loss = negative_log_likelihood(u_pred_w_bias, uncertainty, u, var_coeff)

        #  Regularisation term
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

        loss += args.l2_lambda * l2_norm

        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        epoch_train_loss += batch_loss
        print(f"Batch {batch_idx} Loss : {batch_loss}")

    scheduler.step()
    print(f"Epoch {epoch}/{args.n_epochs} | Total Loss {epoch_train_loss}")
    save_checkpoint(model, optimizer, scheduler, epoch, args.checkpoint_path)

    # TODO : evaluation
