import torch
import torch.nn.functional as F

## cfg
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def l2norm(t):
    return F.normalize(t, dim = -1)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x_shape):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = len(x_shape)
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x_shape[1], x_shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x_shape)]
    return freqs_cis.view(*shape)
