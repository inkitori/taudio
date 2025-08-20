import numpy as np
import torch
from scipy.stats import beta


def beta_medians(n):
    # input: scalar n
    values = np.arange(n) + 1
    x = beta.median(values, n + 1 - values)
    return x


def linear_spline(z, x, y, z_len, x_len):
    # this is basically a numpy function
    # typically, z = beta medians * cumulative hazard, x = cumulative hazard values, y = time
    # x and y are paired, so x_len == y_len
    batch_size = z.shape[0]
    result = np.zeros_like(z)

    for i in range(batch_size):
        # Extract the valid, unpadded data for the current batch item
        valid_x = np.concatenate(([0], x[i, 0:x_len[i]]))
        valid_y = np.concatenate(([0], y[i, 0:x_len[i]]))
        valid_z = z[i, 0:z_len[i]]

        # Perform the linear interpolation
        interpolated_values = np.interp(valid_z, valid_x, valid_y)

        # Place the interpolated values into the result matrix
        result[i, 0:z_len[i]] = interpolated_values

    return torch.tensor(result).to(x.dtype)

# loss function


def poisson_loss(log_hazard, label_mask, frame_mask):
    '''
    log_hazard (batch, seq len): outputs of the model
    label_mask (batch, seq len): boolean mask for when the events occurred, 1 if the event occurred in that frame, 0 otherwise
    frame_mask (batch, seq len): boolean mask for the frame padding
    '''
    cumulative_hazard = torch.sum(torch.exp(log_hazard) * frame_mask, dim=1)
    return cumulative_hazard - (log_hazard * label_mask).sum(dim=1)

# perform inference


def infer_timestamps(n_pred, log_hazards):
    # input: scalar (number of predictions), np.array (log-hazard values)
    hazards = np.exp(log_hazards)
    total_hazard = np.sum(hazards)
    medians = beta_medians(n_pred)

    # return the inverse cumulative hazard
    return np.interp(medians * total_hazard, np.cumsum(np.insert(hazards, 0, 0)), np.arange(log_hazards.shape[0]+1))


def poisson_count_loss(log_hazard, counts, frame_mask):
    '''
    log_hazard (batch, seq len): outputs of the model
    counts (batch, ): boolean mask for when the events occurred, 1 if the event occurred in that frame, 0 otherwise
    frame_mask (batch, seq len): boolean mask for the frame padding
    '''
    cumulative_hazard = torch.cumsum(torch.exp(log_hazard) * frame_mask, dim=1)[:, -1]
    return cumulative_hazard - torch.log(cumulative_hazard) * counts + torch.lgamma(counts + 1)


def infer_count(log_hazard, frame_mask):
    '''
    log_hazard (batch, seq len): outputs of the model
    frame_mask (batch, seq len): boolean mask for the frame padding
    '''
    return torch.floor(torch.cumsum(torch.exp(log_hazard) * frame_mask, dim=1)[:, -1])
