import numpy as np
import torch
from scipy.stats import beta
from accelerate import PartialState
from scipy.signal import find_peaks, windows

import numpy as np

def joint_mode_dynamic_programming(n_pred, log_hazards, min_dist):
    """
    Finds the sequence of 'n_pred' indices that maximizes the sum of log_hazards,
    subject to the constraint that indices must be at least 'min_dist' apart.
    
    Mathematically: argmax_{t1...tn} sum(log_hazards[ti]) s.t. |ti - tj| >= min_dist
    """
    T = len(log_hazards)
    n = int(n_pred)
    
    # DP Table: dp[k, t] stores the max score of placing 'k' events 
    # such that the k-th event is EXACTLY at time 't'.
    # initialized to -infinity
    dp = np.full((n, T), -np.inf)
    
    # Backpointers to reconstruct the path
    # parent[k, t] stores the index of the (k-1)th event 
    parent = np.full((n, T), -1, dtype=int)
    
    # --- Initialization (First Event) ---
    # The first event can be anywhere (that leaves room for n-1 others, technically, 
    # but for simplicity we allow anywhere and mask later)
    dp[0, :] = log_hazards
    
    # --- Recursion ---
    for k in range(1, n):
        # We need to find max(dp[k-1, :]) where index <= t - min_dist
        # To do this efficiently in O(T), we track the running maximum.
        
        running_max_val = -np.inf
        running_max_idx = -1
        
        # Valid range for the previous event ends at t - min_dist
        # We iterate t from 0 to T. The "lookback" index is t - min_dist.
        for t in range(T):
            lookback_idx = t - min_dist
            
            # Update the running max with the new valid candidate from the previous row
            if lookback_idx >= 0:
                val = dp[k-1, lookback_idx]
                if val > running_max_val:
                    running_max_val = val
                    running_max_idx = lookback_idx
            
            # If we have a valid previous event, add current hazard
            if running_max_val > -np.inf:
                dp[k, t] = running_max_val + log_hazards[t]
                parent[k, t] = running_max_idx
                
    # --- Backtracking (Reconstruct Path) ---
    # 1. Find the best position for the last (n-th) event
    last_k = n - 1
    best_end_idx = np.argmax(dp[last_k, :])
    
    # If valid path not found (e.g., sequence too short for n events)
    if dp[last_k, best_end_idx] == -np.inf:
        return np.array([]) 

    best_sequence = np.zeros(n, dtype=int)
    current_idx = best_end_idx
    
    for k in range(n - 1, -1, -1):
        best_sequence[k] = current_idx
        current_idx = parent[k, current_idx]
        
    return best_sequence

def find_n_peaks(signal, n, distance=1, min_height=1e-3):
    """
    Guarantees exactly 'n' peaks are returned.
    Prioritizes:
    1. Peaks satisfying distance & min_height.
    2. Peaks satisfying distance only.
    3. Highest remaining values (ignoring distance).
    """
    n = int(n)
    
    # --- Stage 1: Strict Constraints ---
    # We allow peaks at height >= min_height
    peaks, _ = find_peaks(signal, height=min_height, distance=distance)
    
    if len(peaks) == n:
        return np.sort(peaks)
    
    if len(peaks) > n:
        # Too many peaks: Keep the top N by intensity
        peak_intensities = signal[peaks]
        # argsort gives ascending, we take last n for descending
        top_indices = np.argsort(peak_intensities)[-n:]
        return np.sort(peaks[top_indices])
    
    # --- Stage 2: Relaxed Height (Constraint: Distance) ---
    # If we have < n, we might have missed small valid peaks.
    # Relax height to 0, but keep distance constraint.
    # Note: We assume inputs are probabilities/hazards >= 0.
    peaks_relaxed, _ = find_peaks(signal, height=0, distance=distance)
    
    if len(peaks_relaxed) >= n:
        peak_intensities = signal[peaks_relaxed]
        top_indices = np.argsort(peak_intensities)[-n:]
        return np.sort(peaks_relaxed[top_indices])

    # --- Stage 3: Desperation (Constraint: None) ---
    # If we STILL have < n, it means the 'distance' constraint makes it 
    # mathematically impossible to find N peaks (signal too short / tolerance too wide).
    # We must satisfy the N requirement, so we fill with best remaining points.
    
    # Start with the peaks we found in Stage 2 (max possible with distance constraint)
    final_peaks = list(peaks_relaxed)
    known_peak_set = set(peaks_relaxed)
    
    # Sort ALL indices by intensity (descending)
    all_sorted_indices = np.argsort(signal)[::-1]
    
    for idx in all_sorted_indices:
        if len(final_peaks) >= n:
            break
        if idx not in known_peak_set:
            final_peaks.append(idx)
            known_peak_set.add(idx)
            
    return np.sort(np.array(final_peaks))

def run_iterative_mode(n_pred, log_hazards, kernel, window_width, resmooth=True):
    hazards = np.exp(log_hazards)
    
    half_window = window_width // 2
    
    predictions = []

    if resmooth == False:
        smoothed = np.convolve(hazards, kernel, mode='same')
    
    for _ in range(int(n_pred)):
        if resmooth == True:
            smoothed = np.convolve(hazards, kernel, mode='same')
        
        peak_idx = np.argmax(smoothed)
        
        predictions.append(peak_idx)
        
        start = max(0, peak_idx - half_window)
        end = min(len(hazards), peak_idx + half_window + 1)
        
        if resmooth == True:
            hazards[start:end] = 0.0
        else:
            smoothed[start:end] = 0.0
        
    return np.sort(np.array(predictions))

def run_iterative_posterior_mode(n_pred, hazards, kernel, window_width):
    half_window = window_width // 2
    
    predictions = []
    
    # seq_lens needed for posterior_mode_inhomogeneous_poisson
    seq_lens = np.array([len(hazards)])
    
    # Work on a copy so we can suppress events
    current_hazards = hazards.copy()
    
    for _ in range(int(n_pred)):
        # 1. Smooth the hazards in linear space (re-smooth each iteration)
        smoothed_hazards = np.convolve(current_hazards, kernel, mode='same')
        
        # 2. Convert to log space safely (same as infer_timestamps does)
        smoothed_log_hazards = np.log(smoothed_hazards + 1e-12)
        
        # 3. Call posterior_mode_inhomogeneous_poisson with n_pred=1
        # It expects shape (batch_size, max_seq_len)
        mode_result = posterior_mode_inhomogeneous_poisson(
            n_pred=np.array([1]),  # Finding 1 event at a time
            log_hazards=smoothed_log_hazards[np.newaxis, :],  # Add batch dimension
            seq_lens=seq_lens
        )
        
        # Extract the single prediction (shape is (batch_size, max_n))
        peak_idx = mode_result[0, 0]  # First batch, first prediction
        
        predictions.append(peak_idx)
        
        # 4. Suppression - zero out the window around the prediction
        # peak_idx might be a float, so use its integer part for the window
        peak_int = int(round(peak_idx))
        start = max(0, peak_int - half_window)
        end = min(len(hazards), peak_int + half_window + 1)
        
        # Zero out the current hazards so next iteration finds a different peak
        current_hazards[start:end] = 0.0
    
    return np.sort(np.array(predictions))


def get_beta_mode_fraction(k, n):
    """
    Computes the mode of the Beta distribution in [0, 1].
    Mode = (k - 1) / (n - 1)
    """
    denominator = np.maximum(n - 1, 1e-9)
    mode = (k - 1) / denominator
    return mode

def compute_log_scores(u_candidates, log_hazards, k, n):
    """
    Computes the log-likelihood score for candidate points u.
    Score = log(lambda) + (k-1)log(u) + (n-k)log(1-u)
    """
    alpha = k - 1
    beta = n - k
    
    epsilon = 1e-12
    u_safe = np.clip(u_candidates, epsilon, 1.0 - epsilon)
    
    log_u = np.log(u_safe)
    log_1_u = np.log(1.0 - u_safe)
    
    beta_log_prob = alpha * log_u + beta * log_1_u
    
    return log_hazards + beta_log_prob

def posterior_mode_inhomogeneous_poisson(n_pred, log_hazards, seq_lens):
    """
    Vectorized computation of posterior modes in frame units.
    """
    batch_size, max_seq_len = log_hazards.shape
    max_n = int(np.max(n_pred))
    
    # Masking
    seq_mask = np.arange(max_seq_len)[None, :] < seq_lens[:, None]
    masked_log_hazards = log_hazards.copy()
    masked_log_hazards[~seq_mask] = -np.inf
    
    hazards = np.exp(masked_log_hazards)
    hazards[~seq_mask] = 0.0 

    # Cumulative Hazard F(t)
    cum_hazard_right = np.cumsum(hazards, axis=1)
    H_total = cum_hazard_right[:, -1:] 
    F_right = np.divide(cum_hazard_right, H_total, where=H_total>0)
    F_left = np.pad(F_right[:, :-1], ((0, 0), (1, 0)), mode='constant', constant_values=0)
    
    # Grids
    k_grid = np.arange(1, max_n + 1).reshape(1, 1, -1) # (1, 1, K)
    n_grid = n_pred.reshape(batch_size, 1, 1)          # (B, 1, 1)
    
    F_left_exp = F_left[:, :, np.newaxis]
    F_right_exp = F_right[:, :, np.newaxis]
    log_haz_exp = masked_log_hazards[:, :, np.newaxis]

    # Local Best Candidates
    u_star = get_beta_mode_fraction(k_grid, n_grid) 
    u_best_local = np.clip(u_star, F_left_exp, F_right_exp)
    
    # Evaluate Scores
    scores = compute_log_scores(u_best_local, log_haz_exp, k_grid, n_grid)
    best_frame_indices = np.argmax(scores, axis=1) 
    
    # Transform Back
    b_idx = np.arange(batch_size)[:, None]
    win_F_left = F_left[b_idx, best_frame_indices]     
    win_F_right = F_right[b_idx, best_frame_indices]   
    
    k_idx_3d = np.arange(max_n)[None, :]
    win_u = u_best_local[b_idx, best_frame_indices, k_idx_3d]
    
    F_width = np.maximum(win_F_right - win_F_left, 1e-9)
    local_fraction = (win_u - win_F_left) / F_width
    
    modes = best_frame_indices + local_fraction
    
    valid_mask = k_grid.squeeze() <= n_pred[:, None]
    modes = np.where(valid_mask, modes, np.nan)
    
    return modes

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


import numpy as np
from scipy.signal import windows
import numpy as np
from scipy.signal import windows

import numpy as np
from scipy.signal import windows

def run_iterative_bayesian_greedy(n_pred, log_hazards, tolerance_ms, frame_ms, kernel):
    """
    Combines:
    1. Posterior Mode (Detection): Finds the best frame.
    2. Posterior Median (Refinement): Finds the exact time within that frame's window.
    3. Greedy Suppression: Prevents double-counting.
    """
    outputs = []
    
    # Work on a copy of log_hazards so we can suppress events
    # We maintain 'current_log_hazards' for detection
    current_log_hazards = log_hazards.copy()
    
    # Pre-calculate window size for the refinement/suppression
    # (Same logic as before)
    M = int((tolerance_ms * 2) / frame_ms)
    if M % 2 == 0: M += 1
    half_window = M // 2

    # Pre-smooth hazards for the DETECTION step (Mode finding).
    # We do this once upfront if we want "Strategy B" style detection,
    # OR we can do it iteratively if we want "Strategy A" style.
    # Let's do it iteratively to be safe with suppression.
    
    for _ in range(int(n_pred)):
        # --- STEP 1: DETECTION (Posterior Mode for N=1) ---
        
        # We need the hazards for the current state
        # (Convert to linear space for smoothing, then back to log for stability)
        current_hazards = np.exp(current_log_hazards)
        
        # Apply Smoothing (Convolution)
        # This acts as the "Likelihood Search" over the window
        smoothed_search_hazards = np.convolve(current_hazards, kernel, mode='same')
        
        # Find the Global Maximum (Posterior Mode for N=1)
        # We use argmax because for N=1, the Beta prior is Uniform,
        # so Mode == Argmax of Intensity.
        global_peak_idx = np.argmax(smoothed_search_hazards)
        
        # Stop condition: If probability mass is negligible
        if smoothed_search_hazards[global_peak_idx] < 1e-9:
            break

        # --- STEP 2: REFINEMENT (Local Posterior Median) ---
        
        # Define the window around the mode
        start = max(0, global_peak_idx - half_window)
        end = min(len(current_hazards), global_peak_idx + half_window + 1)
        
        local_mass = current_hazards[start:end]
        total_local_mass = np.sum(local_mass)
        
        if total_local_mass > 0:
            # We want the Median (50% quantile) of THIS local event.
            # Your 'posterior_median' code uses: 
            # np.interp(target_mass, cumsum, indices)
            
            local_cumsum = np.cumsum(local_mass)
            
            # The target is the median of the probability mass in this window
            target = total_local_mass * 0.5
            
            # Perform the interpolation (exactly as in Method C, but local)
            # We insert 0 at the start to handle the left-edge case properly
            local_offset = np.interp(
                target, 
                np.insert(local_cumsum, 0, 0), 
                np.arange(len(local_mass) + 1)
            )
            
            # The interpolation returns an index 0..len(local_mass).
            # 0.5 corresponds to the center of the first frame (index 0).
            # We need to shift it to be 0-indexed relative to 'start'.
            # Note: np.interp on range(0..N+1) with padding effectively gives 
            # the "continuous index".
            # e.g., if mass is in first bin, interp gives 0.5.
            # We subtract 0.5 to center it on the integer frame index.
            
            refined_idx = start + local_offset - 0.5
        else:
            refined_idx = global_peak_idx
            
        outputs.append(refined_idx)
        
        # --- STEP 3: SUPPRESSION ---
        # Mask out this window with -inf (log(0))
        current_log_hazards[start:end] = -np.inf

    return np.sort(np.array(outputs))

def infer_timestamps(n_pred, log_hazards):
    outputs = {}

    # frame_ms = 10
    # tolerance_ms = 40

    # hazards = np.exp(log_hazards)
    
    # # 1. Define the Window Size
    # # We want a window of +/- 100ms (Total 200ms)
    # # If frame is 40ms, 200/40 = 5 frames.
    # window_frames = int((tolerance_ms * 2) / frame_ms)
    
    # # Ensure window is odd so it has a distinct center
    # if window_frames % 2 == 0:
    #     window_frames += 1
        
    # half_window = window_frames // 2
    
    # # Create a convolution kernel to find 'Mass within Window'
    # kernel = np.ones(window_frames)
    
    # # We work on a copy so we can "suppress" peaks as we find them
    # search_probs = hazards.copy()
    
    # predicted_indices = []

    # for _ in range(int(n_pred)):
    #     # 2. Convolve to find the region with Maximum Accuracy
    #     # (The window with the highest sum of probability)
    #     # Using 'same' keeps the indices aligned with the original array
    #     window_sums = np.convolve(search_probs, kernel, mode='same')
        
    #     # Find the center index of the best window
    #     # This maximizes P(hit) within tolerance
    #     global_peak_idx = np.argmax(window_sums)
        
    #     # If the max probability is effectively 0, stop (no more events found)
    #     if window_sums[global_peak_idx] < 1e-9:
    #         break

    #     # 3. Local Refinement: Minimize L1 Loss (Local Median)
    #     # We look strictly inside the winning window
    #     start = max(0, global_peak_idx - half_window)
    #     end = min(len(hazards), global_peak_idx + half_window + 1)
        
    #     local_mass = hazards[start:end]
        
    #     # Calculate Local Median within this specific window
    #     # (Normalize so it sums to 1, then find 0.5 crossing)
    #     local_cumsum = np.cumsum(local_mass)
    #     total_local_mass = local_cumsum[-1]
        
    #     if total_local_mass > 0:
    #         # searchsorted finds the first index where value >= target
    #         median_offset = np.searchsorted(local_cumsum, total_local_mass * 0.5)
    #         refined_idx = start + median_offset
    #     else:
    #         refined_idx = global_peak_idx

    #     predicted_indices.append(refined_idx)
        
    #     # 4. Suppression (The "Greedy" step)
    #     # Zero out the mass in this window so the next iteration 
    #     # finds the *next* highest peak, not the same one.
    #     # We suppress the full window area.
    #     search_probs[start:end] = 0

    # # Return sorted indices (as floats)
    # outputs['default'] = np.sort(np.array(predicted_indices))
    
    # Configuration
    frame_ms = 10
    n_pred_val = np.atleast_1d(n_pred)
    seq_lens = np.array([len(log_hazards)])
    hazards = np.exp(log_hazards)
    total_hazard = np.sum(hazards)
    medians = beta_medians(n_pred)

    # -------------------------------------------------------------------------
    # STRATEGY: Raw Find Peaks (The "Human Eye" Strategy)
    # -------------------------------------------------------------------------
    # This detects the "natural" modes you saw (e.g., 31 or 43 count).
    # height=1e-3: Ignores numerical noise (0.00 vs 0.00001).
    # distance=1: Allows peaks to be right next to each other (indices 12, 13).
    # outputs["raw_find_peaks"] = find_n_peaks(
    #     hazards, 
    #     n=n_pred, 
    #     distance=1,       # Allow neighbors
    #     min_height=1e-3   # Ignore tiny noise
    # ).astype(float)
    
    # 1. Base Predictions (No smoothing)
    outputs["posterior_mode"] = posterior_mode_inhomogeneous_poisson(
        n_pred=n_pred_val, 
        log_hazards=log_hazards[np.newaxis, :], 
        seq_lens=seq_lens
    ).flatten()

    outputs['posterior_median'] = np.interp(
        medians * total_hazard, 
        np.cumsum(np.insert(hazards, 0, 0)), 
        np.arange(log_hazards.shape[0]+1)
    )

    def run_greedy_search(current_hazards, kernel, n, window_width_override=None, interpolate=False):
            search_probs = current_hazards.copy()
            
            # 1. Decouple Kernel Size from Refinement/Suppression Window
            K = len(kernel)
            M = window_width_override if window_width_override is not None else K
            
            half_window = M // 2
            predicted_indices = []

            for _ in range(int(n)):
                # 2. Convolve (Search)
                window_sums = np.convolve(search_probs, kernel, mode='same')
                peak_idx = np.argmax(window_sums)
                
                if window_sums[peak_idx] < 1e-9:
                    break

                # 3. Local Refinement (Median)
                # Now we look at the full window 'M' even if kernel was size 1
                start = max(0, peak_idx - half_window)
                end = min(len(current_hazards), peak_idx + half_window + 1)
                
                local_mass = current_hazards[start:end]
                local_cumsum = np.cumsum(local_mass)
                total_local_mass = local_cumsum[-1]
                
                if total_local_mass > 0:
                    # Standard integer median frame
                    median_idx_local = np.searchsorted(local_cumsum, total_local_mass * 0.5)
                    
                    # OPTIONAL: Linear Interpolation for True Sub-frame Float Precision
                    # If you want 10.4 instead of 10:
                    if interpolate:
                        prev_cum = local_cumsum[median_idx_local-1] if median_idx_local > 0 else 0.0
                        curr_cum = local_cumsum[median_idx_local]
                        mass_at_idx = curr_cum - prev_cum
                        frac = (total_local_mass * 0.5 - prev_cum) / max(1e-9, mass_at_idx)
                        refined_idx = start + median_idx_local + np.clip(frac - 0.5, -0.5, 0.5)
                    else:
                        refined_idx = start + median_idx_local
                else:
                    refined_idx = peak_idx

                predicted_indices.append(refined_idx)
                
                # 4. Suppression
                # Crucial: We now suppress the full 'M' width, preventing double-counting
                search_probs[start:end] = 0
                
            return np.sort(np.array(predicted_indices))

    # --- 3. Sweeping through kernels and tolerances ---
    for tolerance_ms in range(10, 101, 10):
        M = int((tolerance_ms * 2) / frame_ms)
        if M % 2 == 0: M += 1
            
        kernels = {
            "boxcar": np.ones(M),
            # "triang": windows.triang(M) * (M / np.sum(windows.triang(M))),
            # "gauss":  windows.gaussian(M, std=M/4) * (M / np.sum(windows.gaussian(M, std=M/4)))
        }

        for k_name, kernel in kernels.items():
            key_base = f"smooth_{tolerance_ms}ms_{k_name}"
            
            # # A. Strategy: Your original Greedy Search with Suppression
            # # (Calculates local median during the process)
            # outputs[f"{key_base}_greedy_suppression"] = run_greedy_search(
            #     hazards, kernel, n_pred
            # )
            
            # outputs[f"{key_base}_greedy_suppression_interpolate"] = run_greedy_search(
            #     hazards, kernel, n_pred, interpolate=True
            # )

            # # B. Strategy: Posterior Mode (Geometric Smoothing)
            # smoothed_log_hazards = np.convolve(log_hazards, kernel, mode='same')
            # mode_indices = posterior_mode_inhomogeneous_poisson(
            #     n_pred=n_pred_val, 
            #     log_hazards=smoothed_log_hazards[np.newaxis, :], 
            #     seq_lens=seq_lens
            # ).flatten()
            # outputs[f"{key_base}_buggy_posterior_mode"] = mode_indices

            # # C. Strategy: Global Posterior Median (Arithmetic Smoothing)
            # avg_kernel = kernel / np.sum(kernel)
            # smoothed_hazards = np.convolve(hazards, avg_kernel, mode='same')
            # total_smoothed_hazard = np.sum(smoothed_hazards)
            # outputs[f"{key_base}_posterior_median_smoothed"] = np.interp(
            #     medians * total_smoothed_hazard, 
            #     np.cumsum(np.insert(smoothed_hazards, 0, 0)), 
            #     np.arange(log_hazards.shape[0]+1)
            # )

            # ----------------------------------------------------------------
            # B. Strategy: Posterior Mode (FIXED MATH)
            # ----------------------------------------------------------------
            # 1. Convolve in LINEAR space first
            smoothed_hazards_lin = np.convolve(hazards, kernel, mode='same')
            # smoothed_hazards_lin_avg = np.convolve(hazards, avg_kernel, mode='same')
            
            # 2. Convert to log space safely
            smoothed_log_hazards = np.log(smoothed_hazards_lin + 1e-12)
            # smoothed_log_hazards_avg = np.log(smoothed_hazards_lin_avg + 1e-12)
            
            mode_indices = posterior_mode_inhomogeneous_poisson(
                n_pred=n_pred_val, 
                log_hazards=smoothed_log_hazards[np.newaxis, :], 
                seq_lens=seq_lens
            ).flatten()
            outputs[f"{key_base}_fixed_posterior_mode"] = mode_indices

            # outputs[f"{key_base}_find_peaks"] = find_n_peaks(
            #     smoothed_hazards_lin, 
            #     n=n_pred, 
            #     distance=M, 
            #     min_height=1e-3
            # ).astype(float)

            # outputs[f"{key_base}_find_peaks_double_window"] = find_n_peaks(
            #     smoothed_hazards_lin, 
            #     n=n_pred, 
            #     distance=2*M, 
            #     min_height=1e-3
            # ).astype(float)

            # outputs[f"{key_base}_find_peaks_no_smooth"] = find_n_peaks(
            #     hazards, 
            #     n=n_pred, 
            #     distance=M, 
            #     min_height=1e-3
            # ).astype(float)

            # outputs[f"{key_base}_find_peaks_double_window_no_smooth"] = find_n_peaks(
            #     hazards, 
            #     n=n_pred, 
            #     distance=2*M, 
            #     min_height=1e-3
            # ).astype(float)

            # mode_indices_avg = posterior_mode_inhomogeneous_poisson(
            #     n_pred=n_pred_val, 
            #     log_hazards=smoothed_log_hazards_avg[np.newaxis, :], 
            #     seq_lens=seq_lens
            # ).flatten()
            # outputs[f"{key_base}_fixed_avg_posterior_mode"] = mode_indices_avg

            # # ----------------------------------------------------------------
            # # D. NEW Strategy: Fully Smoothed Greedy
            # # ----------------------------------------------------------------
            # # Search on Smoothed AND Refine on Smoothed.
            # # We pass the pre-smoothed hazards.
            # # We pass a kernel of [1] because the hazards are ALREADY smoothed.
            # # If we passed 'kernel' again, we would double-smooth (convolve twice).
            
            # smoothed_hazards_for_greedy = np.convolve(hazards, kernel, mode='same')
            
            # outputs[f"{key_base}_fully_smoothed_greedy"] = run_greedy_search(
            #     smoothed_hazards_for_greedy, 
            #     np.array([1.0]), # Identity kernel, just find peaks in pre-smoothed data
            #     n_pred,
            #     window_width_override=M
            # )

            # outputs[f"{key_base}_fully_smoothed_greedy_argmax"] = run_greedy_search(
            #     smoothed_hazards_for_greedy, 
            #     np.array([1.0]), # Identity kernel, just find peaks in pre-smoothed data
            #     n_pred,
            # )

            # # interpolated greedy

            # outputs[f"{key_base}_fully_smoothed_greedy_interpolate"] = run_greedy_search(
            #     smoothed_hazards_for_greedy, 
            #     np.array([1.0]), # Identity kernel, just find peaks in pre-smoothed data
            #     n_pred,
            #     window_width_override=M,
            #     interpolate=True
            # )

            # outputs[f"{key_base}_fully_smoothed_greedy_argmax_interpolate"] = run_greedy_search(
            #     smoothed_hazards_for_greedy, 
            #     np.array([1.0]), # Identity kernel, just find peaks in pre-smoothed data
            #     n_pred,
            #     interpolate=True
            # )

            # # E. Strategy: Iterative Bayesian Greedy
            # # Exact Posterior Mode (Detection) -> Exact Posterior Median (Refinement)
            # outputs[f"{key_base}_iterative_bayesian"] = run_iterative_bayesian_greedy(
            #     n_pred=n_pred,
            #     log_hazards=log_hazards, # Pass RAW log hazards
            #     tolerance_ms=tolerance_ms,
            #     frame_ms=frame_ms,
            #     kernel=kernel # Pass the smoothing kernel
            # )

            # # ----------------------------------------------------------------
            # # F. STRATEGY: Iterative Re-Smoothing Greedy (Your Request)
            # # ----------------------------------------------------------------
            # # "Searches for mode, zeros region, smooths again"
            outputs[f"{key_base}_iterative_resmoothing"] = run_iterative_mode(
                n_pred=n_pred,
                log_hazards=log_hazards,
                kernel=kernel,
                window_width=M
            )

            # outputs[f"{key_base}_iterative_no_resmoothing"] = run_iterative_mode(
            #     n_pred=n_pred,
            #     log_hazards=log_hazards,
            #     kernel=kernel,
            #     window_width=M,
            #     resmooth=False
            # )

            outputs[f"{key_base}_iterative_resmoothing_posterior_mode"] = run_iterative_posterior_mode(
                n_pred=n_pred,
                hazards=hazards,
                kernel=kernel,
                window_width=M,
            )

            # outputs[f"{key_base}_dp_smoothed"] = joint_mode_dynamic_programming(
            #     n_pred=n_pred,
            #     log_hazards=smoothed_log_hazards,
            #     min_dist=M
            # )

            # outputs[f"{key_base}_dp_unsmoothed"] = joint_mode_dynamic_programming(
            #     n_pred=n_pred,
            #     log_hazards=log_hazards,
            #     min_dist=M
            # )
    
    outputs['default'] = outputs['posterior_mode']

    return outputs

from scipy.special import expit  
def infer_timestamps_binary(n_pred, logits):
    outputs = {}
    frame_ms = 10

    probs = expit(logits)

    outputs['argmax'] = np.argpartition(probs, -n_pred)[-n_pred:]
    outputs['argmax_fixed'] = outputs['argmax'] + 0.5
    outputs['default'] = outputs['argmax_fixed']

    for tolerance_ms in range(10, 101, 10):
        M = int((tolerance_ms * 2) / frame_ms)
        if M % 2 == 0: M += 1
            
        kernels = {
            "boxcar": np.ones(M) / M, # dividing by M shouldnt have any effect
        }

        for k_name, kernel in kernels.items():
            key_base = f"smooth_{tolerance_ms}ms_{k_name}"
            smoothed_probs = np.convolve(probs, kernel, mode='same')

            top_indices = np.argpartition(smoothed_probs, -n_pred)[-n_pred:]
            outputs[key_base] = top_indices + 0.5
            outputs[f"{key_base}_broken"] = top_indices

    return outputs

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
    