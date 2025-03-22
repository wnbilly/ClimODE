from model_utils import ResidualBlock, SelfAttnConv
from torchdiffeq import odeint as odeint
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClimateResnet2D(nn.Module):

    def __init__(self, num_channels, layers, hidden_size):
        super().__init__()
        self.block = ResidualBlock

        conv_layers = []
        for idx in range(len(layers)):
            if idx == 0:
                conv_layers.append(self.make_layer(self.block, num_channels, hidden_size[idx], layers[idx]))
            else:
                conv_layers.append(self.make_layer(self.block, hidden_size[idx - 1], hidden_size[idx], layers[idx]))

        self.conv_layers = nn.ModuleList(conv_layers)

    def make_layer(self, block, in_channels, out_channels, reps):

        layers = [block(in_channels, out_channels)]

        for i in range(1, reps):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()

        for layer in self.conv_layers:
            x = layer(x)

        return x


class ClimateEncoderFreeUncertain(nn.Module):

    def __init__(self, in_channels, const_channels, out_channels, ode_solver, use_attention, use_uq, gamma=0.1):
        super().__init__()
        self.layers = [5, 3, 2]
        self.hidden = [128, 64, 2 * out_channels]
        self.hours_in_a_year = 365 * 24

        self.in_channels = in_channels  #  = K in the paper
        self.out_channels = out_channels
        self.const_channels = const_channels  #  = Amount of static variables (oro and lsm in the paper)

        # phi(x) of Equation (8)
        positional_embedding_channels = 6
        # time_embedding + Equation (7)
        temporal_embedding_channels = 4  # Equation (7) : day(2), season(2)
        spatiotemporal_embedding_channels = positional_embedding_channels * temporal_embedding_channels  # phi(t) X phi(x)

        velocity_network_input_channels = 3 * self.in_channels  # u, v_x, v_y
        velocity_network_input_channels += 2 * self.in_channels  # nabla_u
        velocity_network_input_channels += positional_embedding_channels
        velocity_network_input_channels += temporal_embedding_channels  # + 1  # + 1 as time is also input to the velocity network
        # velocity_network_input_channels += spatiotemporal_embedding_channels
        velocity_network_input_channels += const_channels + 2  # constants channels (oro and lsm in the paper) + phi(h) and phi(w) of Equation (9)

        self.velocity_conv_network = ClimateResnet2D(velocity_network_input_channels, self.layers, self.hidden)

        self.use_attention = use_attention
        if use_attention:
            self.velocity_attention_network = SelfAttnConv(velocity_network_input_channels, 2 * self.in_channels)
            self.gamma = nn.Parameter(torch.tensor([gamma]))

        # odeint method
        self.ode_solver = ode_solver

        #  Use Uncertainty Quantification
        self.use_uq = use_uq
        if use_uq:
            emission_input_channels = self.in_channels
            emission_input_channels += const_channels + 2  # + 2 for radians_lat_map and radians_lon_map
            emission_input_channels += positional_embedding_channels
            emission_input_channels += temporal_embedding_channels
            # emission_input_channels += spatiotemporal_embedding_channels

            self.emission_network = ClimateResnet2D(emission_input_channels, [3, 2, 2], [128, 64, 2 * out_channels])

        # TODO : change this and the processing of constants so any amount of constants can be used
        self.lat_map = None
        self.lon_map = None
        self.const_map = torch.zeros(const_channels)
        self.lsm = None  #  Static variable in the data (cf. Section 3.5)
        self.oro = None  #  Static variable in the data (cf. Section 3.5)

    def set_constants(self, constants_map, lat_map, lon_map):
        self.const_map = constants_map
        self.lat_map = lat_map
        self.lon_map = lon_map
        self.H, self.W = lat_map.shape[-2:]
        self._compute_positional_embedding()

    #  Based on https://towardsdatascience.com/how-to-encode-periodic-time-features-7640d9b21332
    def get_temporal_embedding_for_emission_model(self, t):
        #  t is supposed to be elapsed_hours since the beginning of the year
        #  t of shape (batch_size, n_timesteps)

        day_vector = 2 * torch.pi * t / 24  # to mimic day cycles
        sin_day_emb = torch.sin(day_vector)
        cos_day_emb = torch.cos(day_vector)

        year_vector = 2 * torch.pi * t / (24 * 365)  #  to mimic year cycles
        sin_year_emb = torch.sin(year_vector)
        cos_year_emb = torch.cos(year_vector)

        time_embedding = torch.stack([cos_day_emb, sin_day_emb, cos_year_emb, sin_year_emb], dim=-1)

        #  From : https://github.com/pytorch/pytorch/issues/9410
        return time_embedding[:, :, :, None, None].expand(-1, -1, -1, self.H, self.W)

    def get_temporal_embedding_for_pde(self, t, is_batched=False):
        #  t is supposed to be elapsed_hours since the beginning of the year
        #  t of shape ()

        day_vector = 2 * torch.pi * t / 24  # to mimic day cycles
        sin_day_emb = torch.sin(day_vector)
        cos_day_emb = torch.cos(day_vector)

        year_vector = 2 * torch.pi * t / (24 * 365)  #  to mimic year cycles
        sin_year_emb = torch.sin(year_vector)
        cos_year_emb = torch.cos(year_vector)

        time_embedding = torch.stack([cos_day_emb, sin_day_emb, cos_year_emb, sin_year_emb], dim=-1)

        #  From : https://github.com/pytorch/pytorch/issues/9410
        if is_batched:
            return time_embedding[:, :, None, None].expand(-1, -1, self.H, self.W)
        else:
            return time_embedding[None, :, None, None].expand(-1, -1, self.H, self.W)

    def _compute_positional_embedding(self):
        # Converting to radians
        radians_lat_map = torch.deg2rad(self.lat_map).unsqueeze(0)
        radians_lon_map = torch.deg2rad(self.lon_map).unsqueeze(0)
        # Expand after computation
        cos_lat_map, sin_lat_map = torch.cos(radians_lat_map), torch.sin(radians_lat_map)
        cos_lon_map, sin_lon_map = torch.cos(radians_lon_map), torch.sin(radians_lon_map)

        # shape (6, H, W)
        self.positional_embedding = torch.cat(
            [cos_lat_map, cos_lon_map, sin_lat_map, sin_lon_map, sin_lat_map * cos_lon_map, sin_lat_map * sin_lon_map]
        )

    def get_positional_embedding(self, batch_size=1):
        # shape (batch_size, 6, H, W)
        return self.positional_embedding.expand(batch_size, -1, self.H, self.W)

    #  Section 3.3 : 2nd order PDE as a system of 1st order ODEs
    # Modified to fit torchode. See https://torchode.readthedocs.io/en/latest/nd-data/
    def torchpde(self, t, u_v, positional_embedding, radians_lat_map, radians_lon_map):

        u_v = u_v.view(-1, 3 * self.in_channels, self.H, self.W)
        batch_size = u_v.shape[0]

        # u_v is u, v_x, v_y concatenated (along the dim=1)
        u, v_x, v_y = u_v.split(self.in_channels, 1)
        grad_u_x = torch.gradient(u, dim=2)[0]
        grad_u_y = torch.gradient(u, dim=3)[0]
        nabla_u = torch.cat([grad_u_x, grad_u_y], dim=1)

        t *= self.hours_in_a_year  # To retrieve the initial time scale
        temporal_embedding = self.get_temporal_embedding_for_pde(t, is_batched=True)

        # TODO : compute part of comb_rep out of self.pde ?
        # Equation (8)
        # removed spatio_temporal_embedding as I think it is redundant (and it takes 24 channels)
        #  spatiotemporal_embedding = self.get_spatiotemporal_embedding(temporal_embedding, positional_embedding)
        constants = torch.cat([positional_embedding, radians_lat_map, radians_lon_map, self.lsm, self.oro], dim=1).expand(batch_size, -1, -1, -1)
        comb_rep = torch.cat([u, nabla_u, v_x, v_y, temporal_embedding, constants], dim=1)

        #  Section 3.4 : Modeling local and global effects
        #  dv is the derivate wrt t of the flow velocity v
        if self.use_attention:
            dv = self.velocity_conv_network(comb_rep) + self.gamma * self.velocity_attention_network(comb_rep)
        else:
            dv = self.velocity_conv_network(comb_rep)

        #  Equation (2)
        transport_term = v_x * grad_u_x + v_y * grad_u_y  # v.nabla_u
        compression_term = u * (torch.gradient(v_x, dim=2)[0] + torch.gradient(v_y, dim=3)[0])  # u.nabla_v
        du = transport_term + compression_term

        #  Derivatives of equation (5)
        derivatives = torch.cat([du, dv], dim=1)
        return derivatives.flatten(start_dim=1)

    #  Section 3.3 : 2nd order PDE as a system of 1st order ODEs
    def pde(self, t, u_v, positional_embedding, radians_lat_map, radians_lon_map):

        # u_v is u, v_x, v_y concatenated (along the dim=1)
        u, v_x, v_y = u_v.split(self.in_channels, 1)
        grad_u_x = torch.gradient(u, dim=2)[0]
        grad_u_y = torch.gradient(u, dim=3)[0]
        nabla_u = torch.cat([grad_u_x, grad_u_y], dim=1)

        t *= self.hours_in_a_year  # To retrieve the proper time scale
        temporal_embedding = self.get_temporal_embedding_for_pde(t)

        # TODO : compute part of comb_rep out of self.pde ?
        # Equation (8)
        # removed spatio_temporal_embedding as I think it is redundant (and it takes 24 channels)
        #  spatiotemporal_embedding = self.get_spatiotemporal_embedding(temporal_embedding, positional_embedding)
        comb_rep = torch.cat(
            [u, nabla_u, v_x, v_y, temporal_embedding, positional_embedding, radians_lat_map, radians_lon_map, self.lsm, self.oro], dim=1
        )

        #  Section 3.4 : Modeling local and global effects
        #  dv is the derivate wrt t of the flow velocity v
        if self.use_attention:
            dv = self.velocity_conv_network(comb_rep) + self.gamma * self.velocity_attention_network(comb_rep)
        else:
            dv = self.velocity_conv_network(comb_rep)

        #  Equation (2)
        transport_term = v_x * grad_u_x + v_y * grad_u_y  # v.nabla_u
        compression_term = u * (torch.gradient(v_x, dim=2)[0] + torch.gradient(v_y, dim=3)[0])  # u.nabla_v
        du = - transport_term - compression_term

        #  Derivatives of equation (5)
        derivatives = torch.cat([du, dv], dim=1)
        return derivatives

    def get_spatiotemporal_embedding(self, temporal_embedding, positional_embedding):
        return torch.cat([temporal_embedding[:, idx].unsqueeze(dim=1) * positional_embedding for idx in range(temporal_embedding.shape[1])], dim=1)

    #  Section 3.7 : System sources and uncertainty estimation
    def emission_model(self, t, constants_and_positional_embedding, u_pred):
        batch_size, n_timesteps = t.shape[:2]

        # To pass each step of each batch one by one in the emission model
        artificial_batch_size = batch_size * n_timesteps

        temporal_embedding = self.get_temporal_embedding_for_emission_model(t).flatten(0, 1)
        constants_and_positional_embedding = constants_and_positional_embedding.expand(artificial_batch_size, -1, -1, -1)

        # spatiotemporal_embedding = self.get_spatiotemporal_embedding(temporal_embedding, positional_embedding)
        comb_rep = torch.cat(
            [u_pred.flatten(0, 1), temporal_embedding, constants_and_positional_embedding], dim=1
        )

        # out is of shape (batch_size*n_years, 2*n_quantities, self.H, self.W) is mean and std of u
        #  self.emission_network takes predicted steps of u as input (shape of (n_ode_steps, self.H, self.W) * batch_size)
        out = self.emission_network(comb_rep).view(batch_size, n_timesteps, 2 * self.out_channels, self.H, self.W)

        mean = u_pred + out[:, :, :self.out_channels]
        std = F.softplus(out[:, :, self.out_channels:])

        return mean, std

    def forward(self, u_0, v_0, t, delta_t, atol=0.1, rtol=0.1):
        batch_size, n_timesteps = t.shape[:2]

        # velocity samples + quantities data
        # v_0 of shape (batch_size, K, 2, H, W), u_0 of shape (batch_size, K, H, W)
        v_0_u_0 = torch.cat([u_0, v_0[:, :, 0], v_0[:, :, 1]], dim=1)  #  shape (batch_size, 3*K, H, W)

        #  TODO : make the amount of constants modifiable
        self.oro, self.lsm = self.const_map[0, 0], self.const_map[0, 1]
        self.lsm = self.lsm[None, None, :]
        self.oro = F.normalize(self.oro)[None, None, :]

        # Converting to radians
        radians_lat_map = torch.deg2rad(self.lat_map).unsqueeze(0).expand(1, -1, self.H, self.W)
        radians_lon_map = torch.deg2rad(self.lon_map).unsqueeze(0).expand(1, -1, self.H, self.W)

        positional_embedding = self.get_positional_embedding()

        # make the ODE forward function
        ode_func = lambda t_, uv_: self.pde(t_, uv_, positional_embedding, radians_lat_map, radians_lon_map)
        u_v_predicted = torch.zeros((batch_size, n_timesteps, 3 * self.in_channels, self.H, self.W), device=u_0.device)

        # TODO : odeint batched
        # NOTE : Could use torchode (https://github.com/martenlienen/torchode) to directly process a batch
        for idx in range(batch_size):
            init_time = t[idx, 0].item()  #  * self.hours_in_a_year  #  time in hours
            final_time = t[idx, -1].item()  # * self.hours_in_a_year
            # ODE evaluation steps : filled the gaps time gaps
            # I don't know if it is useful as odeint already adds sampling steps
            integration_steps = torch.linspace(init_time, final_time, steps=n_timesteps * delta_t, dtype=torch.float32, device=u_0.device)
            integration_steps /= self.hours_in_a_year  #  To have smaller time steps for the integration
            u_v_pred_integrated = odeint(ode_func, v_0_u_0[idx:idx + 1], integration_steps, method=self.ode_solver, atol=atol, rtol=rtol)
            # Keep value every delta_t h step
            u_v_predicted[idx] = u_v_pred_integrated[0::delta_t].swapaxes(0, 1)

        #  with torchode, based on https://torchode.readthedocs.io/en/latest/torchdiffeq/
        '''
        # import torchode as to # to place with the imports
        ode_func = lambda t_, uv_: self.torchpde(t_, uv_, positional_embedding, radians_lat_map, radians_lon_map)
        term = to.ODETerm(ode_func)
        step_method = to.Euler(term=term)
        step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=term)
        adjoint = to.AutoDiffAdjoint(step_method, step_size_controller)
        integration_steps = torch.stack(
            [torch.linspace(t_[0].item(), t_[-1].item(), steps=n_timesteps * delta_t, dtype=torch.float32, device=u_0.device) for t_ in t]
            )
        problem = to.InitialValueProblem(y0=v_0_u_0.flatten(start_dim=1), t_eval=integration_steps)
        sol = adjoint.solve(problem)

        abs_err = (u_v_pred_integrated - sol).abs()
        print(f"mean err : {abs_err.mean()}     max err : {abs_err.max()}")'''

        #  Extract u out (u, vx, vy)
        u_predicted = u_v_predicted[:, :, :self.in_channels, :, :]

        if self.use_uq:
            constants_and_positional_embedding = torch.cat(
                [positional_embedding, radians_lat_map, radians_lon_map, self.lsm, self.oro], dim=1
            )
            mean, std = self.emission_model(t, constants_and_positional_embedding, u_predicted)

        else:
            mean, std = torch.zeros_like(u_predicted), torch.zeros_like(u_predicted)

        return mean, std, u_predicted
