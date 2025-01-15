from torch.nn import functional as F, Module, init
import math
from utils import *
import torch
from torch import Tensor
from torch.nn.parameter import Parameter




class ComplexLinearFirst(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    --zhanglei:
    --We constructed a neural network with real numbers as inputs and complex numbers as both intermediate neurons and outputs.
    --Within each neuron, matrix multiplication follows the rules of complex number multiplication.
    --In this process, since the inputs are real numbers, we duplicated the real part and used it to fill the imaginary part to prevent any delay in training the imaginary part.
    --This layer is only applicable to the first layer, where the input is real and the output is complex.
    Examples::

        # >>> m = nn.Linear(20, 30)
        # >>> input = torch.randn(128, 20)
        # >>> output = m(input)
        # >>> print(output.size())
        torch.Size([128, 60])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_A = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight_B = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias_c = Parameter(torch.empty(out_features, **factory_kwargs))
            self.bias_d = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_A, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_B, a=math.sqrt(5))
        if self.bias_c is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_A)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias_c, -bound, bound)
            init.uniform_(self.bias_d, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        complex_input = torch.cat((input, input), dim=1)
        complex_weights = torch.cat(
            (torch.cat((self.weight_A, self.weight_B), dim=1), torch.cat((-1 * self.weight_B, self.weight_A), dim=1)),
            dim=0)
        complex_bias = torch.cat((self.bias_c, self.bias_d), dim=0)
        return F.linear(complex_input, complex_weights, complex_bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias_c is not None
        )


class ComplexLinearMidden(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    --zhanglei:
    --A neural network with complex numbers as inputs, complex numbers as intermediate neurons, and complex numbers as outputs.
    --Within each neuron, matrix multiplication follows the rules of complex number multiplication.

    # >>> m = nn.Linear(20, 30)
        # >>> input = torch.randn(128, 40)
        40 is because it includes R part and I part
        # >>> output = m(input)
        # >>> print(output.size())
        torch.Size([128, 60])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_A = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight_B = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias_c = Parameter(torch.empty(out_features, **factory_kwargs))
            self.bias_d = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight_A, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_B, a=math.sqrt(5))
        if self.bias_c is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_A)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias_c, -bound, bound)
            init.uniform_(self.bias_d, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        complex_input = input
        complex_weights = torch.cat(
            (torch.cat((self.weight_A, self.weight_B), dim=1), torch.cat((-1 * self.weight_B, self.weight_A), dim=1)),
            dim=0)
        complex_bias = torch.cat((self.bias_c, self.bias_d), dim=0)
        return F.linear(complex_input, complex_weights, complex_bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias_c is not None
        )


if __name__ == '__main__':
    m = ComplexLinearFirst(2, 3)
    input = torch.randn(5, 2)
    output = m(input)
    print('output.size():', output.size())
    print(input)
    print(output)

    mm = ComplexLinearMidden(3, 1)
    outputt = mm(output)
    print('outputt.size():', outputt.size())
    print(outputt)
