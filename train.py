from torch.utils.data import DataLoader
import argparse
import torch.optim as optim
import torch
import os

# local imports
from models import ClimateEncoderFreeUncertain
from utils import set_seed, negative_log_likelihood, load_checkpoint, save_checkpoint, verif_path, custom_loss
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
parser.add_argument('--n_steps', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--l2_lambda', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--data_path', type=str, default='era_5_data')
parser.add_argument('--model_save_folder', type=str, default='checkpoints')
parser.add_argument('--checkpoint_path', type=str, default='checkpoints/checkpoint.pt')
parser.add_argument('--velocity_data_path', type=str)
parser.add_argument('--validation_velocity_data_path', type=str)
parser.add_argument('--device', type=str, default='auto')
args = parser.parse_args()

#  args processing
verif_path(args.model_save_folder)
verif_path(os.path.dirname(args.checkpoint_path))

levels = ["z", "t", "t2m", "u10", "v10"]
data_folders = ['geopotential_500/*.nc', 'temperature_850/*.nc', '2m_temperature/*.nc', '10m_u_component_of_wind/*.nc',
                '10m_v_component_of_wind/*.nc']
constants = ['orography', 'lsm']
const_folders = ['constants/constants_5.625deg.nc']
paths_to_data = [os.path.join(args.data_path, folder) for folder in data_folders]
const_info_path = [os.path.join(args.data_path, folder) for folder in const_folders]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else torch.device(args.device)
print(f"device : {device}")

# Training data
# The timestamps of the given velocity file are the reference timestamps
velocity, timestamps = load_velocity(args.velocity_data_path)
min_date, max_date = timestamps[0], timestamps[-1]

data_infos = [get_resampled_normalized_data(level_path, str(min_date), str(max_date), levels[idx]) for idx, level_path in enumerate(paths_to_data)]
data = torch.cat([data_info[0] for data_info in data_infos], dim=1)
data_scales = {'min':torch.tensor([data_info[2] for data_info in data_infos]), 'max':torch.tensor([data_info[3] for data_info in data_infos])}
del data_infos

# Validation data
val_velocity, val_timestamps = load_velocity(args.validation_velocity_data_path)
val_min_date, val_max_date = val_timestamps[0], val_timestamps[-1]

# Load the validation data with the training scales for the min max scaling
val_data_infos = [get_resampled_normalized_data(level_path, str(val_min_date), str(val_max_date), levels[idx],scales=[data_scales['min'][idx].item(), data_scales['max'][idx].item()]) for idx, level_path in enumerate(paths_to_data)]
val_data = torch.cat([data_info[0] for data_info in val_data_infos], dim=1)
del val_data_infos

# Constants
constants_to_fetch = ['orography', 'lsm']
# const_channels_info : tensor of shape (1, n_constants, H, W)
const_channels_info, lat_map, lon_map = fetch_constant_info(const_info_path, constants_to_fetch)

n_constants = len(constants_to_fetch)
n_quantities = len(paths_to_data)  # K in the paper
H, W = data.shape[2], data.shape[3]  #  H : number of points latitude-wise, W : number of points longitude-wise

dataset = utvDataset(data, velocity, timestamps, n_steps=args.n_steps, window_step=args.n_steps)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.custom_collate_fn)

val_dataset = utvDataset(val_data, val_velocity, val_timestamps, n_steps=args.n_steps, window_step=args.n_steps)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.custom_collate_fn)

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
    model, optimizer, scheduler, start_epoch = load_checkpoint(args.checkpoint_path, model, optimizer, scheduler)
    start_epoch += 1

for epoch in range(start_epoch, args.n_epochs):

    model.train()

    epoch_train_loss = .0
    epoch_val_loss = .0
    best_val_loss = torch.inf

    # RMSD = []

    if epoch == 0:
        var_coeff = 0.001
    else:
        var_coeff = 2 * scheduler.get_last_lr()[0]

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        u = batch['u'].to(device)
        u0 = u[:, 0].to(device)
        t = batch['t'].to(device)
        v0 = batch['v0'].to(device)

        u_pred_w_bias, uncertainty, u_pred = model(u0, v0, t, args.delta_t)

        loss = negative_log_likelihood(u_pred_w_bias, uncertainty, u, var_coeff)
        #  Regularisation term
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

        loss += args.l2_lambda * l2_norm

        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        epoch_train_loss += batch_loss
        print(f"Batch {batch_idx} loss : {batch_loss}")

    scheduler.step()
    save_checkpoint(model, optimizer, scheduler, epoch , args.checkpoint_path)

    model.eval()

    for batch_idx, batch in enumerate(val_dataloader):
        optimizer.zero_grad()

        u = batch['u'].to(device)
        u0 = u[:, 0].to(device)
        t = batch['t'].to(device)
        v0 = batch['v0'].to(device)

        u_pred_w_bias, uncertainty, u_pred = model(u0, v0, t, args.delta_t)

        loss = negative_log_likelihood(u_pred_w_bias, uncertainty, u, var_coeff)
        #  Regularisation term
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

        loss += args.l2_lambda * l2_norm

        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        epoch_val_loss += batch_loss

    print(f"-> epoch {epoch}/{args.n_epochs} | train loss {epoch_train_loss:.5f} | val loss : {epoch_val_loss:.5f}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_epoch = epoch
        save_checkpoint(model, optimizer, scheduler, epoch, args.checkpoint_path.replace('.pt', f'_best.pt'), verbose=True)
