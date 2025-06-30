import numpy as np
import torch

def beta_medians(n_samples):
    # input: (batch size,)
    batch_size = n_samples.shape[0]
    n_samples = n_samples.detach().cpu().numpy()
    max_len = int(np.max(n_samples))
    mask = np.arange(max_len)[None, :] < n_samples[:, None]
    values = np.tile(np.arange(max_len)[:,None], batch_size).T
    values[~mask] = 0.0
    values = values + 1
    x = beta.median(values, n_samples[:,None] + 1 - values)
    return torch.Tensor(x * mask)

def linear_spline(z, x, y, z_len, x_len):
    # this is basically a numpy function
    # typically, z = beta medians * cumulative hazard, x = cumulative hazard values, y = time
    # x and y are paired, so x_len == y_len
    batch_size = z.shape[0]
    result = np.zeros_like(z)

    for i in range(batch_size):
        # Extract the valid, unpadded data for the current batch item
        valid_x = np.concatenate( ([0], x[i, 0:x_len[i]]) )
        valid_y = np.concatenate( ([0], y[i, 0:x_len[i]]) )
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
    cumulative_hazard = torch.cumsum(torch.exp(log_hazard) * frame_mask, dim=1)[:, -1]
    return -(-cumulative_hazard + (log_hazard * label_mask).sum(dim=1))

# perform inference
def infer_timestamps(num_pred, log_hazard, frame_mask, num_frames):
    '''
    num_pred (batch,): number of events per batch, i.e., how many timestamp predictions do you need per batch?
    log_hazard (batch, seq len): outputs of the model
    frame_mask (batch, seq len): boolean mask for the frame padding
    num_frames (batch,): number of audio frames per batch (it should just be the sum of the frame_mask along the batch axis)
    
    The prediction will be the index of the frame. Note that the outputs are on the cpu, not gpu!
    '''
    n_batch = num_pred.shape[0]
    max_seq_len = log_hazard.shape[1]
    
    medians = beta_medians(num_pred)
    hazards = torch.exp(log_hazard) * frame_mask
    cumulative_hazard = torch.cumsum(hazards)
    inv_cum_haz = linear_spline(
        (medians * cumulative_hazard[:, -1]).cpu().numpy(),
        cumulative_hazard.cpu().numpy(),
        np.tile(np.arange(max_seq_len) + 1, (k, 1)),
        num_pred.cpu().numpy(),
        num_frames.cpu().numpy())
    return torch.floor(torch.tensor(inv_cum_haz))
