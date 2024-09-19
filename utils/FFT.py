import numpy as np
import torch
import torch.nn.functional as F


def compute_fft_of_weights(weight_tensor):
    # Flatten the weight tensor if it's more than 2D
    if len(weight_tensor.shape) > 2:
        weight_tensor = weight_tensor.view(weight_tensor.shape[0], -1)

    if len(weight_tensor.shape) == 2:
        # For 2D tensors, apply 2D FFT
        fft_weights = np.fft.fft2(weight_tensor.cpu().numpy(), axes=[0, 1])
    elif len(weight_tensor.shape) == 1:
        # For 1D tensors, apply 1D FFT
        fft_weights = np.fft.fft(weight_tensor.cpu().numpy())
    else:
        raise ValueError("Unsupported tensor shape for FFT: {}".format(weight_tensor.shape))

    return fft_weights.flatten()


def FFT_weights_cal(model):
    # List to store all FFT-transformed weights
    all_fft_weights = []

    # Iterate through layers and compute FFT for layers with weights
    for name, layer in model.named_modules():
        if hasattr(layer, 'weight') and layer.weight is not None:
            weight_tensor = layer.weight.data
            fft_weights = compute_fft_of_weights(weight_tensor)
            all_fft_weights.extend(fft_weights)

    # Convert the list to a numpy array
    all_fft_weights_array = np.array(all_fft_weights)
    return all_fft_weights_array


def spectral_cal(model, input=True):
    F_weights = FFT_weights_cal(model)
    prob_dist = np.abs(F_weights)
    prob_dist /= np.sum(prob_dist)

    # prob_dist = torch.tensor(prob_dist, dtype=torch.float)
    #
    # if input == True:
    #     prob_dist = F.log_softmax(prob_dist, dim=0)
    return prob_dist
