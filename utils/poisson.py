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
    # # input: scalar (number of predictions), np.array (log-hazard values)
    # hazards = np.exp(log_hazards)
    # total_hazard = np.sum(hazards)
    # medians = beta_medians(n_pred)

    # # return the inverse cumulative hazard
    # return np.interp(medians * total_hazard, np.cumsum(np.insert(hazards, 0, 0)), np.arange(log_hazards.shape[0]+1))
    """
    Infer timestamps using the Mode (Peak Finding) strategy.
    
    Args:
        n_pred (int): Number of events to predict.
        log_hazards (np.array): Shape (seq_len,), log intensity values.
        
    Returns:
        np.array: Shape (n_pred,), sorted timestamps (indices).
    """
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
    """
    Infers timestamps by finding the highest accuracy windows (peaks),
    then minimizing L1 loss locally within those windows.
    
    Args:
        n_pred (int): Number of timestamps to predict.
        log_hazards (np.array): Log-hazard (or log-probability) values.
        frame_ms (float): Duration of one frame in ms (default 40).
        tolerance_ms (float): The tolerance radius for accuracy (default 100).
                              Total window size will be +/- tolerance.
                              
    Returns:
        np.array: Predicted timestamp indices (float).
    """
    
    # ARGMAX STRATEGY

    # best_idx = int(np.argmax(log_hazards))
    # # match previous format: 1-D float array of timestamps
    # return np.array([best_idx], dtype=float)

    # NON OVERLAPPING WINDOWS STRATEGY

    # frame_ms = 10
    # tolerance_ms = 50

    # hazards = np.exp(log_hazards)
    
    # # 1. Define the Window Size (Fixed Frame Size)
    # # Total window is 2x tolerance (e.g., 100ms)
    # window_frames = int((tolerance_ms * 2) / frame_ms)
    
    # # (Optional) Ensure window is odd if you prefer, 
    # # though for fixed tiling even numbers are fine.
    # if window_frames % 2 == 0:
    #     window_frames += 1

    # # 2. Reshape into Non-Overlapping Windows
    # # We truncate the end of the array to make it divisible by window_frames
    # n_windows = len(hazards) // window_frames
    # trunc_len = n_windows * window_frames
    
    # # Create a view of the array as distinct blocks
    # # Shape becomes: (Number of Windows, Frames per Window)
    # tiled_hazards = hazards[:trunc_len].reshape(n_windows, window_frames)
    
    # # 3. Calculate Mass for every window at once
    # # Sum along axis 1 (summing mass inside each window)
    # window_sums = tiled_hazards.sum(axis=1)
    
    # # 4. Select the Top 'n_pred' Windows
    # # We filter for the requested number of events, ensuring we don't exceed available windows
    # count_to_pick = min(int(n_pred), n_windows)
    
    # # argsort gives indices of windows sorted by mass (ascending), 
    # # we take the last 'count_to_pick' and reverse to get descending order
    # top_window_indices = np.argsort(window_sums)[-count_to_pick:][::-1]
    
    # predicted_indices = []

    # for w_idx in top_window_indices:
    #     # If the window mass is effectively 0, skip
    #     if window_sums[w_idx] < 1e-9:
    #         continue

    #     # 5. Local Refinement: Minimize L1 Loss (Local Median)
    #     # We work directly with the specific row in our tiled matrix
    #     local_mass = tiled_hazards[w_idx]
        
    #     local_cumsum = np.cumsum(local_mass)
    #     total_local_mass = local_cumsum[-1]
        
    #     # Find the offset within this specific window where we cross 50% mass
    #     median_offset = np.searchsorted(local_cumsum, total_local_mass * 0.5)
        
    #     # 6. Map back to Global Index
    #     # Global Index = (Window Index * Window Size) + Local Offset
    #     global_idx = (w_idx * window_frames) + median_offset
        
    #     predicted_indices.append(global_idx)

    # # Return sorted indices
    # return np.sort(np.array(predicted_indices))

    # OVERLAPPING WINDOWS STRATEGY

    frame_ms = 10
    tolerance_ms = 100

    hazards = np.exp(log_hazards)
    
    # 1. Define the Window Size
    # We want a window of +/- 100ms (Total 200ms)
    # If frame is 40ms, 200/40 = 5 frames.
    window_frames = int((tolerance_ms * 2) / frame_ms)
    
    # Ensure window is odd so it has a distinct center
    if window_frames % 2 == 0:
        window_frames += 1
        
    half_window = window_frames // 2
    
    # Create a convolution kernel to find 'Mass within Window'
    kernel = np.ones(window_frames)
    
    # We work on a copy so we can "suppress" peaks as we find them
    search_probs = hazards.copy()
    
    predicted_indices = []

    for _ in range(int(n_pred)):
        # 2. Convolve to find the region with Maximum Accuracy
        # (The window with the highest sum of probability)
        # Using 'same' keeps the indices aligned with the original array
        window_sums = np.convolve(search_probs, kernel, mode='same')
        
        # Find the center index of the best window
        # This maximizes P(hit) within tolerance
        global_peak_idx = np.argmax(window_sums)
        
        # If the max probability is effectively 0, stop (no more events found)
        if window_sums[global_peak_idx] < 1e-9:
            break

        # 3. Local Refinement: Minimize L1 Loss (Local Median)
        # We look strictly inside the winning window
        start = max(0, global_peak_idx - half_window)
        end = min(len(hazards), global_peak_idx + half_window + 1)
        
        local_mass = hazards[start:end]
        
        # Calculate Local Median within this specific window
        # (Normalize so it sums to 1, then find 0.5 crossing)
        local_cumsum = np.cumsum(local_mass)
        total_local_mass = local_cumsum[-1]
        
        if total_local_mass > 0:
            # searchsorted finds the first index where value >= target
            median_offset = np.searchsorted(local_cumsum, total_local_mass * 0.5)
            refined_idx = start + median_offset
        else:
            refined_idx = global_peak_idx

        predicted_indices.append(refined_idx)
        
        # 4. Suppression (The "Greedy" step)
        # Zero out the mass in this window so the next iteration 
        # finds the *next* highest peak, not the same one.
        # We suppress the full window area.
        search_probs[start:end] = 0

    # Return sorted indices (as floats)
    return np.sort(np.array(predicted_indices))

    # OVERLAPPING WINDOWS TRIANGULAR KERNEL

    # frame_ms = 10
    # tolerance_ms = 50

    # hazards = np.exp(log_hazards)

    # # 1. Define the Window Size
    # window_frames = int((tolerance_ms * 2) / frame_ms)

    # # Ensure window is odd so it has a distinct center
    # if window_frames % 2 == 0:
    #     window_frames += 1
        
    # half_window = window_frames // 2

    # # --- MODIFIED SECTION START ---
    # # Create a Triangular Kernel
    # # Instead of np.ones(window_frames), we create a ramp up and down.
    # # For a window of 5, this creates: [1, 2, 3, 2, 1]
    # ramp_up = np.arange(1, half_window + 2)
    # ramp_down = np.arange(half_window, 0, -1)
    # kernel = np.concatenate((ramp_up, ramp_down))

    # # Optional: Normalize the kernel so the peak value reflects probability magnitude
    # # (Not strictly necessary for argmax, but good for debugging)
    # # kernel = kernel / kernel.sum()
    # # --- MODIFIED SECTION END ---

    # search_probs = hazards.copy()

    # predicted_indices = []

    # for _ in range(int(n_pred)):
    #     # 2. Convolve to find the region with Maximum Weighted Accuracy
    #     window_sums = np.convolve(search_probs, kernel, mode='same')
        
    #     global_peak_idx = np.argmax(window_sums)
        
    #     if window_sums[global_peak_idx] < 1e-9:
    #         break

    #     # 3. Local Refinement: Minimize L1 Loss (Local Median)
    #     start = max(0, global_peak_idx - half_window)
    #     end = min(len(hazards), global_peak_idx + half_window + 1)
        
    #     local_mass = hazards[start:end]
        
    #     local_cumsum = np.cumsum(local_mass)
    #     total_local_mass = local_cumsum[-1]
        
    #     if total_local_mass > 0:
    #         median_offset = np.searchsorted(local_cumsum, total_local_mass * 0.5)
    #         refined_idx = start + median_offset
    #     else:
    #         refined_idx = global_peak_idx

    #     predicted_indices.append(refined_idx)
        
    #     # 4. Suppression
    #     # Note: We still suppress the 'box' area to prevent re-detecting the same peak
    #     search_probs[start:end] = 0

    # return np.sort(np.array(predicted_indices))

    # OVERLAPPING WINDOWS GAUSSIAN KERNEL
    # frame_ms = 10
    # tolerance_ms = 50

    # hazards = np.exp(log_hazards)

    # # 1. Define the Window Size
    # window_frames = int((tolerance_ms * 2) / frame_ms)

    # if window_frames % 2 == 0:
    #     window_frames += 1
        
    # half_window = window_frames // 2

    # # --- MODIFIED SECTION START ---
    # # Create a Gaussian Kernel
    # # We define sigma so that the window edges (half_window) are 3 standard deviations away.
    # sigma = half_window / 3.0

    # # Create an array of indices centered at 0: [-2, -1, 0, 1, 2]
    # x = np.linspace(-half_window, half_window, window_frames)

    # # Calculate Gaussian: e^(-x^2 / 2*sigma^2)
    # kernel = np.exp(-0.5 * (x / sigma) ** 2)
    # # --- MODIFIED SECTION END ---

    # search_probs = hazards.copy()

    # predicted_indices = []

    # for _ in range(int(n_pred)):
    #     # 2. Convolve
    #     window_sums = np.convolve(search_probs, kernel, mode='same')
        
    #     global_peak_idx = np.argmax(window_sums)
        
    #     # Check if peak is effectively zero
    #     if window_sums[global_peak_idx] < 1e-9:
    #         break

    #     # 3. Local Refinement: Minimize L1 Loss
    #     start = max(0, global_peak_idx - half_window)
    #     end = min(len(hazards), global_peak_idx + half_window + 1)
        
    #     local_mass = hazards[start:end]
    #     local_cumsum = np.cumsum(local_mass)
    #     total_local_mass = local_cumsum[-1]
        
    #     if total_local_mass > 0:
    #         median_offset = np.searchsorted(local_cumsum, total_local_mass * 0.5)
    #         refined_idx = start + median_offset
    #     else:
    #         refined_idx = global_peak_idx

    #     predicted_indices.append(refined_idx)
        
    #     # 4. Suppression
    #     # We still use 'hard' suppression (zeroing out the box) so we don't 
    #     # find the same peak twice.
    #     search_probs[start:end] = 0

    # return np.sort(np.array(predicted_indices))


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
    