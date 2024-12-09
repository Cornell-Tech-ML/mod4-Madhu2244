from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    # Calculate the new height and width after pooling
    new_height = height // kh
    new_width = width // kw

    # Reshape the input tensor
    # Step 1: Use `view` to split the height and width into blocks of size `kh` and `kw`
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    # Step 2: Permute the dimensions to bring the kernel elements together
    # (batch, channel, new_height, new_width, kh, kw) -> (batch, channel, new_height, new_width, kh * kw)
    tiled = (
        reshaped.permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .view(batch, channel, new_height, new_width, kh * kw)
    )

    return tiled, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling to the input tensor given kernel size.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    batch, channel, _, _ = input.shape
    tiled, new_height, new_width = tile(input, kernel)

    # Average over the last dimension
    return tiled.mean(dim=4).view(batch, channel, new_height, new_width)


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to reduce over

    Returns:
    -------
        Tensor of size batch x channel x height x width where argmax is 1, 0 otherwise (one-hot tensor)

    """
    return input == max_reduce(input, dim)


class Max(Function):
    """A custom autograd function for computing the maximum value along a specified dimension of a tensor."""

    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for the Max function.

        Args:
        ----
            ctx: Context object for storing information for backward computation.
            input: Input tensor for which the maximum reduction will be applied.
            dim: Dimension along which to compute the maximum.

        Returns:
        -------
            Tensor: A tensor containing the maximum values along the specified dimension.

        Notes:
        -----
            - Saves the input tensor and the dimension for use in the backward pass.
            - The `max_reduce` function is used to perform the reduction.

        """
        dim_val = int(dim.item())
        ctx.save_for_backward(input, dim_val)
        return max_reduce(input, dim_val)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for the Max function.

        Args:
        ----
            ctx: Context object containing saved values from the forward pass.
            grad_output: The gradient of the loss with respect to the output of the forward pass.

        Returns:
        -------
            Tuple[Tensor, float]:
                - Tensor: The gradient of the loss with respect to the input tensor.
                - float: Zero gradient for the `dim` parameter (as it is non-learnable).

        Notes:
        -----
            - The gradient is computed using the argmax operation, which identifies
              the indices of the maximum values along the specified dimension.
            - The resulting gradient is scaled by the `grad_output`.

        """
        input, dim = ctx.saved_values
        return grad_output * argmax(input, dim), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Compute the maximum values along a specified dimension of the input tensor.

    Args:
    ----
        input: A Tensor on which the max reduction operation will be performed.
        dim: The dimension along which to compute the maximum.

    Returns:
    -------
        Tensor: A new tensor containing the maximum values along the specified dimension.

    Notes:
    -----
        - This function reduces the input tensor along the given dimension, retaining the maximum value for each slice.
        - It utilizes the `Max` autograd function to enable backpropagation through the operation.

    Example:
    -------
        >>> input = torch.tensor([[1, 3, 2], [4, 0, 5]])
        >>> max(input, dim=1)
        tensor([3, 5])

    """
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int | None = None) -> Tensor:
    """Compute the softmax over the specified dimension.

    Args:
    ----
        input: Tensor on which to compute softmax.
        dim: Dimension to apply softmax. If None, defaults to the last dimension.

    Returns:
    -------
        Tensor after applying softmax.

    """
    exp = input.exp()
    return exp / exp.sum(dim)


def logsoftmax(input: Tensor, dim: int | None = None) -> Tensor:
    """Compute the log-softmax over the specified dimension.

    Args:
    ----
        input: Tensor on which to compute log-softmax.
        dim: Dimension to apply log-softmax. If None, defaults to the last dimension.

    Returns:
    -------
        Tensor after applying log-softmax.

    """
    exp = input.exp()
    return (exp / exp.sum(dim)).log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D max pooling on the input tensor.

    Args:
    ----
        input: Tensor of shape (batch, channel, height, width)
        kernel: Tuple (kernel_height, kernel_width)

    Returns:
    -------
        Tensor of shape (batch, channel, new_height, new_width) after max pooling.

    """
    tiled, new_height, new_width = tile(input, kernel)

    return max(tiled, dim=4).view(input.shape[0], input.shape[1], new_height, new_width)


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Apply dropout to the input tensor.

    Args:
    ----
        input: Tensor to apply dropout.
        p: Probability of dropping an element (0 <= p <= 1).
        ignore: If True, ignore dropout and return the input unchanged.

    Returns:
    -------
        Tensor after applying dropout.

    """
    if ignore or p == 0.0:
        return input  # No dropout applied

    # Mask of 1/0 values based on probability p
    mask = rand(input.shape) > p
    return input * mask
