import os
import numpy as np
import torch
from torch import nn, optim as optim
import torch.nn.functional as F
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from tqdm import tqdm


class OptimVelocity(nn.Module):
    def __init__(self, K, H, W):
        super(OptimVelocity, self).__init__()
        self.v_x = torch.nn.Parameter(torch.randn(K, H, W))
        self.v_y = torch.nn.Parameter(torch.randn(K, H, W))

    def forward(self, u):
        du_y = torch.gradient(u, dim=1)[0]  # (H,W) --> (y,x)
        du_x = torch.gradient(u, dim=2)[0]
        advection = self.v_x * du_x + self.v_y * du_y + u * (torch.gradient(self.v_y, dim=1)[0] + torch.gradient(self.v_x, dim=2)[0])
        return advection, self.v_x, self.v_y


#  Section 3.6 : Initial velocity inference
def optimize_velocity(data, delta_u, vel_model, kernel, K, H, W, device, steps=200, alpha=0.0000001, lr=2):
    model = vel_model(K, H, W).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    loss_step = []
    # Train velocity model
    for step in range(steps):
        optimizer.zero_grad()

        advection, v_x, v_y = model(data)

        # Compute smoothing term
        kernel_v_x = v_x.view(K, -1, 1)
        kernel_v_y = v_y.view(K, -1, 1)
        kernel_expand = kernel.expand(K, kernel.shape[0], kernel.shape[1])
        v_x_kernel = torch.matmul(kernel_v_x.transpose(1, 2), kernel_expand)
        final_x = torch.matmul(v_x_kernel, kernel_v_x).mean()
        v_y_kernel = torch.matmul(kernel_v_y.transpose(1, 2), kernel_expand)
        final_y = torch.matmul(v_y_kernel, kernel_v_y).mean()

        v_loss = F.mse_loss(delta_u, advection) + alpha * (final_x + final_y)
        loss_step.append(v_loss.item())

        if v_loss.item() < best_loss:
            best_loss = v_loss.item()
            best_vx = v_x
            best_vy = v_y
            best_advection = advection

        v_loss.backward()
        optimizer.step()

    return best_vx, best_vy, best_loss, best_advection


def fit_velocity(data, time_sampling, vel_model, kernel, device, n_samples=3, use_pbar=True):
    n_timesteps = data.shape[0]
    K, H, W = data.shape[-3:]
    v_shape = (n_timesteps - (n_samples - 1), 2, K, H, W)  # two derivatives (x and y)
    v = torch.zeros(v_shape)

    if data.device != device:
        data = data.to(device)
    if kernel.device != device:
        kernel = kernel.to(device)

    pbar = tqdm(range(n_samples, n_timesteps), desc='Fitting the velocity ') if use_pbar else range(n_samples, n_timesteps)

    for idx in pbar:
        past_samples = data[idx - n_samples:idx]  # past_samples includes the sample for which the velocity is being computed
        delta_u = get_du_dt(past_samples, time_sampling)
        v_x, v_y, best_loss, out = optimize_velocity(data[idx], delta_u, vel_model, kernel, K, H, W, device)
        v[idx - n_samples, 0] = v_x
        v[idx - n_samples, 1] = v_y
    return v


def get_delta_u(u_vel, t_steps, time_sampling):
    """
    Estimates du/dt using previous time steps
    :param u_vel:
    :param t_steps:
    :param time_sampling:
    :return:
    """
    t = t_steps.flatten().float() * time_sampling
    input_u_vel = u_vel.view(u_vel.shape[0], u_vel.shape[1], -1)
    coeffs = natural_cubic_spline_coeffs(t, input_u_vel)
    spline = NaturalCubicSpline(coeffs)
    point = t[-1]
    out = spline.derivative(point).view(-1, u_vel.shape[2], u_vel.shape[3], u_vel.shape[4])

    return out


def get_du_dt(u_samples, time_sampling):
    """
    Estimates du/dt using previous time steps
    :param u_samples:
    :param time_sampling:
    :return:
    """
    n_samples = u_samples.shape[0]
    t = torch.arange(n_samples, dtype=torch.float, device=u_samples.device) * time_sampling
    input_u_vel = u_samples.view(n_samples, -1)

    coeffs = natural_cubic_spline_coeffs(t, input_u_vel)
    spline = NaturalCubicSpline(coeffs)

    point = t[-1]
    out = spline.derivative(point).view(-1, u_samples.shape[2], u_samples.shape[3])

    return out


def get_gauss_kernel(shape, lat, lon):
    rows, columns = shape
    kernel = torch.zeros(shape[0] * shape[1], shape[0] * shape[1])
    pos = []
    for i in range(rows):
        for j in range(columns):
            pos.append([lat[i], lon[j]])

    for i in range(rows * columns):
        for j in range(rows * columns):
            dist = torch.sum((torch.tensor(pos[i]) - torch.tensor(pos[j])) ** 2)
            kernel[i][j] = torch.exp(-dist / (2 * 1 * 1))

    return torch.linalg.inv(kernel)
