import numpy as np
import torch
from scipy.stats import beta
from accelerate import PartialState

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
    with torch.autocast(device_type=log_hazard.device.type, enabled=False):
        log_hazard = log_hazard.to(torch.float32)
        frame_mask = frame_mask.to(torch.float32)
        label_mask = label_mask.to(torch.float32)

        label_mask = label_mask * frame_mask

        cumulative_hazard = torch.sum(torch.exp(log_hazard) * frame_mask, dim=1)
        loss = cumulative_hazard - (log_hazard * label_mask).sum(dim=1)

        return loss
# perform inference


def infer_timestamps(n_pred, log_hazards):
    # input: scalar (number of predictions), np.array (log-hazard values)
    hazards = np.exp(log_hazards)
    total_hazard = np.sum(hazards)
    medians = beta_medians(n_pred)

    # return the inverse cumulative hazard
    return np.interp(medians * total_hazard, np.cumsum(np.insert(hazards, 0, 0)), np.arange(log_hazards.shape[0]+1))
    # """
    # Infer timestamps using the Mode (Peak Finding) strategy.
    
    # Args:
    #     n_pred (int): Number of events to predict.
    #     log_hazards (np.array): Shape (seq_len,), log intensity values.
        
    # Returns:
    #     np.array: Shape (n_pred,), sorted timestamps (indices).
    # """
    # from scipy.signal import find_peaks
    # n_pred = int(n_pred)
    # if n_pred == 0:
    #     return np.array([], dtype=float)

    # hazards = np.exp(log_hazards)
    
    # # 1. Find local maxima (peaks)
    # # distance=2 ensures we don't pick immediate neighbors (frame t and t+1) 
    # # as separate events. Adjust 'distance' based on your frame resolution.
    # peaks, properties = find_peaks(hazards, height=0, distance=2)
    
    # peak_heights = properties['peak_heights']
    
    # # 2. Select the top N peaks
    # if len(peaks) >= n_pred:
    #     # Get indices of the peaks with the highest hazard values
    #     # specific_indices are indices into the 'peaks' array, not the audio frames
    #     top_peak_indices = np.argsort(peak_heights)[-n_pred:]
    #     selected_timestamps = peaks[top_peak_indices]
        
    # else:
    #     # FALLBACK: If the model predicts N events, but we found fewer than N peaks
    #     # (e.g. the curve is flat or noisy), we fill the remaining spots 
    #     # with the highest raw values from the hazard curve that aren't already peaks.
        
    #     # Take all found peaks
    #     selected_timestamps = peaks.tolist()
        
    #     # Create a mask to exclude indices we already selected
    #     mask = np.ones_like(hazards, dtype=bool)
    #     mask[peaks] = False
        
    #     # How many more do we need?
    #     remaining_count = n_pred - len(peaks)
        
    #     # Get the indices of the highest values from the non-peak areas
    #     # We perform argsort on the masked hazards
    #     masked_hazards = np.where(mask, hazards, -1.0) # Set already picked to -1
    #     remaining_indices = np.argsort(masked_hazards)[-remaining_count:]
        
    #     selected_timestamps.extend(remaining_indices)
    #     selected_timestamps = np.array(selected_timestamps)

    # # 3. Sort by time (timestamps must be sequential) and cast to float
    # # to match the original output type of interpolation
    # return np.sort(selected_timestamps).astype(float)


def poisson_count_loss(log_hazard, counts, frame_mask):
    '''
    log_hazard (batch, seq len): outputs of the model
    counts (batch, ): boolean mask for when the events occurred, 1 if the event occurred in that frame, 0 otherwise
    frame_mask (batch, seq len): boolean mask for the frame padding
    '''
    with torch.autocast(device_type=log_hazard.device.type, enabled=False):
        log_hazard = log_hazard.to(torch.float32)
        frame_mask = frame_mask.to(torch.float32)
        counts = counts.to(torch.float32)

        cumulative_hazard = torch.sum(torch.exp(log_hazard) * frame_mask, dim=1)
        loss = cumulative_hazard - torch.log(cumulative_hazard) * counts + torch.lgamma(counts + 1)

        return loss

def infer_count(log_hazard, frame_mask):
    '''
    log_hazard (batch, seq len): outputs of the model
    frame_mask (batch, seq len): boolean mask for the frame padding
    '''
    with torch.autocast(device_type=log_hazard.device.type, enabled=False):
        log_hazard = log_hazard.to(torch.float32)
        frame_mask = frame_mask.to(torch.float32)

        count = torch.round(torch.sum(torch.exp(log_hazard) * frame_mask, dim=1))

        return count
    