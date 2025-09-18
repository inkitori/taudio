import numpy as np
import torch
from scipy.stats import beta
import logging

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
    # Calculate cumulative hazard with intermediate tensor cleanup
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
    # To use logsumexp with a mask, we can set the masked-out values
    # of log_hazard to a very large negative number (-inf).
    # This ensures their exp() value is 0 and they don't contribute to the sum.
    # Note: .bool() is important if frame_mask is 0/1 floats.
    masked_log_hazard = log_hazard.masked_fill(~frame_mask.bool(), -torch.inf)

    # 1. Calculate the log of the cumulative hazard directly and stably.
    log_cumulative_hazard = torch.logsumexp(masked_log_hazard, dim=1)

    # 2. We still need the cumulative hazard for the first term of the loss.
    # We can get this by exponentiating the stable log_cumulative_hazard.
    # This is safe because logsumexp has already handled potential overflows.
    cumulative_hazard = torch.exp(log_cumulative_hazard)

    # The Poisson Negative Log-Likelihood is:  λ - k*log(λ) + log(k!)
    # where λ is cumulative_hazard and k is counts.
    # We use our stably computed terms here.
    loss = cumulative_hazard - counts * log_cumulative_hazard + torch.lgamma(counts + 1)

    return loss.mean() # Often, we take the mean over the batch.

	# cumulative_hazard = torch.sum(torch.exp(log_hazard) * frame_mask, dim=1)
	# return cumulative_hazard - (log_hazard * counts).sum(dim=1)

    # # Ensure mask is float for multiplication
    # frame_mask_float = frame_mask.float()

    # # 1. Calculate λ (total_expected_count) for each sequence in the batch
    # # hazard = exp(log_hazard)
    # # λ = sum(hazard * mask) over the sequence length dimension
    # total_expected_count = torch.sum(torch.exp(log_hazard) * frame_mask_float, dim=1)

    # # Add a small epsilon for numerical stability in case total_expected_count is zero
    # log_total_expected_count = torch.log(total_expected_count + 1e-8)

    # # 2. Calculate the Negative Log-Likelihood (NLL)
    # # Loss = λ - k * log(λ)
    # # where k is the true `counts`
    # # Ensure counts is float for the multiplication
    # nll = total_expected_count - counts.float() * log_total_expected_count

    # # 3. Return the mean loss over the batch
    # return torch.mean(nll)

    # cumulative_hazard = log_hazard[:, -1] # shape: (batch,)
    # return torch.mean(torch.abs(counts - cumulative_hazard))

    # log_hazard = log_hazard.to(torch.float64)
    # counts = counts.to(torch.float64)
    # counts += 0.5
    # frame_mask = frame_mask.to(torch.float64)
    
    # cumulative_hazard = torch.exp(log_hazard[:, -1])
    # return cumulative_hazard - torch.log(cumulative_hazard) * counts + torch.lgamma(counts + 1)

    # # return mean squared error instead
    # return torch.mean((counts - torch.sum(log_hazard * frame_mask, dim=1)) ** 2)

    # '''
    # A more numerically stable version of the Poisson count loss.

    # log_hazard (batch, seq len): Raw outputs of the model (in log-space).
    # counts (batch, ): The number of events that occurred in each sequence.
    # frame_mask (batch, seq len): Boolean mask for the frame padding (True for valid frames).
    # '''
    # # Ensure frame_mask is a boolean tensor for masking
    # logging.info(f"log_hazard: {log_hazard}")
    # logging.info(f"counts: {counts}")
    # logging.info(f"frame_mask: {frame_mask}")
    # logging.info(f"log_hazard.dtype: {log_hazard.dtype}")
    # logging.info(f"counts.dtype: {counts.dtype}")
    # logging.info(f"frame_mask.dtype: {frame_mask.dtype}")
    # frame_mask = frame_mask.bool()

    # # To correctly use logsumexp with masking, we set the log_hazard of masked-out
    # # frames to -inf. The exp of -inf is 0, so they don't contribute to the sum.
    # masked_log_hazard = log_hazard.masked_fill(~frame_mask, -float('inf'))

    # # Calculate the log of the cumulative hazard in a stable way using the
    # # Log-Sum-Exp trick. This avoids intermediate overflow/underflow from exp().
    # log_cumulative_hazard = torch.logsumexp(masked_log_hazard, dim=1)

    # logging.info(f"log_cumulative_hazard: {log_cumulative_hazard}")

    # # The Poisson negative log-likelihood is: cumulative_hazard - counts * log(cumulative_hazard) + log(k!)
    # # We can rewrite this using our stably computed log_cumulative_hazard.
    # # Note: torch.log(cumulative_hazard) is equivalent to log_cumulative_hazard.
    # cumulative_hazard = torch.exp(log_cumulative_hazard)

    # logging.info(f"cumulative_hazard: {cumulative_hazard}")

    # # Final loss calculation using the stable components.
    # # torch.lgamma(counts + 1) is a stable way to compute log(counts!)
    # loss = cumulative_hazard - counts * log_cumulative_hazard + torch.lgamma(counts + 1)

    # logging.info(f"loss: {loss}")

    # return loss


def infer_count(log_hazard, frame_mask):
    '''
    log_hazard (batch, seq len): outputs of the model
    frame_mask (batch, seq len): boolean mask for the frame padding
    '''
    # Ensure inputs have the same shape
    if not (log_hazard.shape == frame_mask.shape):
        raise ValueError("Input tensors log_hazard and frame_mask must have the same shape.")

    # 1. Convert log_hazard to hazard (which is the expected count per frame)
    hazard = torch.exp(log_hazard)

    # 2. Apply the mask to zero out contributions from padded frames
    masked_hazard = hazard * frame_mask

    # 3. Sum the expected counts over the sequence length dimension (dim=1)
    # This gives the total expected count for each sequence in the batch.
    predicted_total_counts = torch.floor(torch.sum(masked_hazard, dim=1))

    return predicted_total_counts

    # cumulative_hazard = torch.exp(log_hazard[:, -1])
    # return torch.floor(cumulative_hazard)
    # return torch.round(predicted_counts)

    # cumulative_hazard = log_hazard[:, -1] # shape: (batch,)
    # logging.info(f"cumulative_hazard: {cumulative_hazard}")
    # return torch.floor(cumulative_hazard+0.3333-0.02/cumulative_hazard)

    # log_hazard = log_hazard.to(torch.float64)
    # frame_mask = frame_mask.to(torch.float64)
    
    # cumulative_hazard = torch.sum(torch.relu(log_hazard) * frame_mask, dim=1)
