import xarray as xr
from datetime import datetime, timezone
import torch
import numpy as np
from torch.utils.data import Dataset


def dt64_to_long(dt64):
    return dt64.astype(np.long)


def long_to_dt64(long_):
    return np.datetime64(long_, 'us')


class utvDataset(Dataset):
    def __init__(self, states, velocities, timestamps, n_steps):
        """
        Initialize the dataset with states and timestamps.

        Args:
            states (array-like): Array of state data.
            timestamps (array-like): Array of corresponding timestamps.
        """
        assert len(states) == len(timestamps) and len(states) == len(velocities), "States and timestamps must have the same length."
        self.states = states
        self.timestamps = timestamps
        self.velocities = velocities
        self.n_steps = n_steps
        self.seconds_in_an_hour = 3600

    # Based on : https://stackoverflow.com/questions/13703720/converting-between-datetime-timestamp-and-datetime64
    def _dt64_to_dt(self, dt64):
        ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    def normalize_time_over_year(self, timestamp):
        # Normalise time over a year

        if isinstance(timestamp, np.datetime64):
            timestamp = self._dt64_to_dt(timestamp)

        start_of_year = datetime(timestamp.year, 1, 1, tzinfo=timezone.utc)
        end_of_year = datetime(timestamp.year + 1, 1, 1, tzinfo=timezone.utc)

        # Total seconds in the year
        total_hours_in_year = (end_of_year - start_of_year).total_seconds() / self.seconds_in_an_hour

        # Elapsed seconds since the start of the year
        elapsed_hours = (timestamp - start_of_year).total_seconds() / self.seconds_in_an_hour

        return elapsed_hours / total_hours_in_year

    def __len__(self):
        """Return the total number of samples."""
        # Remove self.n_steps because the last items would not make a full sequence
        return len(self.states) - self.n_steps

    def __getitem__(self, idx):
        """
        Retrieve the sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary with 'u', 't', 'v0', 'timestamps'.
        """
        return {
            'u'         : self.states[idx:idx + self.n_steps],  # u
            't'         : torch.tensor([self.normalize_time_over_year(ts) for ts in self.timestamps[idx:idx + self.n_steps]]),
            #  'timestamps': torch.tensor([dt64_to_long(ts.item()) for ts in self.timestamps[idx:idx + self.n_steps]], dtype=torch.long),
            # NOTE : ndarray is used for timestamps to use np.datetime64 as torch does not allow to store datetime types in tensors
            'v0'        : self.velocities[idx],  #  v0
            'timestamps': self.timestamps[idx:idx + self.n_steps]
        }

    def stack_fn(self, array):
        if isinstance(array, np.ndarray):
            return np.stack
        elif isinstance(array, torch.Tensor):
            return torch.stack

    def custom_collate_fn(self, batch):
        return {key_: self.stack_fn(batch[0][key_])([item[key_] for item in batch]) for key_ in batch[0]}


def load_velocity(path, out_type=torch.tensor, device=None):
    velocity_dataset = xr.open_mfdataset(path)
    # swapaxes to have (time, K, 2, H, W)
    return out_type(np.array(velocity_dataset.to_array())).movedim(0,1), velocity_dataset['time'].values


def fetch_constant_info(path, vars_to_fetch=None):
    """

    :param path: path of the dataset to load using xrarray
    :param vars_to_fetch: list of var names to fetch in the loaded dataset. If None, fetch all variables
    :return:
    """
    data = xr.open_mfdataset(path, combine='by_coords')
    if vars_to_fetch is None:
        vars_to_fetch = list(data.data_vars)  # Fetch all the variables

    constants_data = torch.from_numpy(data[vars_to_fetch].to_array().values).unsqueeze(0)

    # TODO : make it more modular for other datasets
    return constants_data, torch.from_numpy(data['lat2d'].values), torch.from_numpy(data['lon2d'].values)


def get_resampled_normalized_data(data_path, min_date, max_date, level, sampling_time=6):
    sampling_time_str = f"{sampling_time}h"
    data = xr.open_mfdataset(data_path, combine='by_coords')
    # data = data.isel(lat=slice(None, None, -1))
    if level in ["v", "u", "r", "q", "tisr"]:
        data = data.sel(level=500)
    data = data.resample(time=sampling_time_str).nearest(tolerance="1h")  # Setting data to be 6-hour cycles
    data_global = data.sel(time=slice(min_date, max_date)).load()

    max_val = data_global.max()[level].values.tolist()
    min_val = data_global.min()[level].values.tolist()

    data_normalised = torch.tensor((data_global[level].values - min_val) / (max_val - min_val)).unsqueeze(1)
    timestamps = data_global.time.values

    return data_normalised, timestamps
