from torch import nn, Tensor
from torch.nn import Parameter, init
import torch
from torch.nn import functional as F
import math
from .grad_buffer import GradBuffer

class DecomposedLinear(nn.Module):
    r"""Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.decompose_linear_function = None

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def prepare_decompose_linear(self, Buffer: GradBuffer):
        class Decomposed_LinearFunction(torch.autograd.Function):
            # Note that both forward and backward are @staticmethods
            @staticmethod
            # bias is an optional argument
            def forward(ctx, input, weight, bias=None):
                #print(input.shape)
                ctx.save_for_backward(input, weight, bias)
                output = torch.matmul(input, weight.t())
                if bias is not None:
                    output += bias.unsqueeze(0).expand_as(output)
                return output

            # This function has only a single output, so it gets only one gradient
            @staticmethod
            def backward(ctx, grad_output):
                # This is a pattern that is very convenient - at the top of backward
                # unpack saved_tensors and initialize all gradients w.r.t. inputs to
                # None. Thanks to the fact that additional trailing Nones are
                # ignored, the return statement is simple even when the function has
                # optional inputs.
                input, weight, bias = ctx.saved_tensors
                grad_input = grad_bias = None

                # These needs_input_grad checks are optional and there only to
                # improve efficiency. If you want to make your code simpler, you can
                # skip them. Returning gradients for inputs that don't require it is
                # not an error.
                #if ctx.needs_input_grad[0]:
                    #grad_input = grad_output.mm(weight)
                #if ctx.needs_input_grad[1]:
                    #grad_weight = grad_output.t().mm(input)
                #if bias is not None and ctx.needs_input_grad[2]:
                    #grad_bias = grad_output.sum(0)
                if ctx.needs_input_grad[0]:
                    grad_input = torch.matmul(grad_output, weight)
                if bias is not None:
                    grad_bias = grad_output.sum(0)

                Buffer.update(input.reshape(-1, input.size(-1)), grad_output.reshape(-1, grad_output.size(-1)))
                return grad_input, None, grad_bias
        self.decompose_linear_function = Decomposed_LinearFunction

    def disable_decompose(self):
        self.decompose_linear_function = None

    def forward(self, input: Tensor) -> Tensor:
        if self.decompose_linear_function is None:
            return F.linear(input, self.weight, self.bias)
        else:
            return self.decompose_linear_function.apply(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"