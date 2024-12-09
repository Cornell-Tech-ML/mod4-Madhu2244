from typing import Tuple, TypeVar, Any

from numba import njit as _njit
import numpy as np
from numba import prange

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Wrapper around numba.njit that sets inline='always' by default.

    Args:
    ----
        fn: Function to compile
        kwargs: Additional arguments to pass to numba.njit

    Returns:
    -------
        Compiled function

    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right

    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    # s1 = input_strides
    # s2 = weight_strides

    # Iterate over all elements in the output tensor
    for out_flat_idx in prange(out_size):
        # Compute the output tensor indices
        out_index = np.empty(3, np.int32)
        to_index(out_flat_idx, out_shape, out_index)
        batch_idx, out_channel_idx, out_width_idx = out_index

        # Initialize accumulator for the output value
        accumulator = 0.0

        # Perform the convolution for each channel and kernel position
        for in_channel_idx in range(in_channels):
            for kernel_idx in range(kw):
                # Compute the input index with respect to the kernel offset
                offset = kernel_idx if not reverse else kw - 1 - kernel_idx
                input_width_idx = (
                    out_width_idx - offset if reverse else out_width_idx + offset
                )

                # Skip if the input index is out of bounds
                if input_width_idx < 0 or input_width_idx >= width:
                    continue

                # Calculate flat indices for input and weight
                input_idx = (
                    batch_idx * input_strides[0]
                    + in_channel_idx * input_strides[1]
                    + input_width_idx * input_strides[2]
                )
                weight_idx = (
                    out_channel_idx * weight_strides[0]
                    + in_channel_idx * weight_strides[1]
                    + kernel_idx * weight_strides[2]
                )

                # Accumulate the convolution result
                accumulator += input[input_idx] * weight[weight_idx]

        # Set the accumulated value in the output tensor
        out[out_flat_idx] = accumulator

    # TODO: Implement for Task 4.1.
    # raise NotImplementedError("Need to implement for Task 4.1")


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
        -------
            batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward for Conv1dFun"""
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right

    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # inners
    # s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    # s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    # Iterate over each element in the output tensor
    for out_flat_idx in prange(out_size):
        # Compute the output tensor indices
        out_index = np.empty(4, np.int32)
        to_index(out_flat_idx, out_shape, out_index)
        batch_idx, out_channel_idx, out_h_idx, out_w_idx = out_index

        # Initialize accumulator for the output value
        accumulator = 0.0

        # Perform the convolution for each input channel and kernel position
        for in_channel_idx in range(in_channels):
            for k_h in range(kh):
                for k_w in range(kw):
                    # Compute the input index with respect to the kernel offset
                    offset_h = k_h if not reverse else kh - 1 - k_h
                    offset_w = k_w if not reverse else kw - 1 - k_w
                    input_h_idx = (
                        out_h_idx - offset_h if reverse else out_h_idx + offset_h
                    )
                    input_w_idx = (
                        out_w_idx - offset_w if reverse else out_w_idx + offset_w
                    )

                    # Skip if the input index is out of bounds
                    if (
                        input_h_idx < 0
                        or input_h_idx >= height
                        or input_w_idx < 0
                        or input_w_idx >= width
                    ):
                        continue

                    # Calculate flat indices for input and weight
                    input_idx = (
                        batch_idx * s1[0]
                        + in_channel_idx * s1[1]
                        + input_h_idx * s1[2]
                        + input_w_idx * s1[3]
                    )
                    weight_idx = (
                        out_channel_idx * s2[0]
                        + in_channel_idx * s2[1]
                        + k_h * s2[2]
                        + k_w * s2[3]
                    )

                    # Accumulate the convolution result
                    accumulator += input[input_idx] * weight[weight_idx]

        # Set the accumulated value in the output tensor
        out[out_flat_idx] = accumulator

    # # TODO: Implement for Task 4.2.
    # raise NotImplementedError("Need to implement for Task 4.2")


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
        -------
            (:class:`Tensor`) : batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward for COnv2dFun"""
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
