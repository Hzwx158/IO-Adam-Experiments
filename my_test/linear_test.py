import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer, AdamW
from typing import Any, Iterable
from functools import wraps

_TS = torch.Tensor

class TensorQueue:
    def __init__(self, q_max:int, *ele_shape) -> None:
        if isinstance(q_max, tuple): #copy
            self.__t, self.__cur_len = q_max
        else:
            assert q_max > 0, f"Expect q_max > 0 but got q_max = {q_max}"
            self.__t = torch.zeros((q_max, *ele_shape))
            self.__cur_len = 0
    
    @property
    def element_shape(self): return self.__t.shape[1:]
    @property
    def max_size(self): return self.__t.shape[0]
    def __len__(self): return self.__cur_len
    def empty(self): return self.__cur_len == 0
    def full(self): return self.__cur_len == self.max_size
    @property
    def front(self): 
        return self.__t[0] if not self.empty() else None
    def pop(self):
        if not self.empty(): 
            self.__t[0:self.__cur_len-1] = self.__t[1:self.__cur_len]
            self.__cur_len -= 1
    def push(self, v:torch.Tensor, auto_pop = False):
        if self.full():
            if not auto_pop: return
            self.pop()
        self.__t[self.__cur_len] = v
        self.__cur_len += 1
    
    def push_group(self, v:Iterable[torch.Tensor], auto_pop = False):
        for t in v:
            self.push(t, auto_pop)

    def __str__(self) -> str:
        return f"TensorQueue(\n\
            q_max = {self.max_size}\n\
            len = {self.__cur_len}\n\
            t = {self.__t}\n)"
    
    def to(self, *args, **kwargs):
        return TensorQueue((
            self.__t.to(*args, **kwargs),
            self.__cur_len
        ))

    @property
    def data(self): return self.__t[0:self.__cur_len]
    def __getitem__(self, idx): 
        return self.__t[idx]
    def __setitem__(self, idx, v):
        self.__t[idx] = v

class QueueParam(nn.Parameter, metaclass = torch.nn.parameter._ParameterMeta):
    eps = 1e-8
    def __new__(cls, q_max = 64, data: torch.Tensor = ..., requires_grad: bool = ...):
        return nn.Parameter.__new__(cls, data, requires_grad)

    def __init__(self, q_max = 64, data: torch.Tensor = ..., requires_grad: bool = ...):
        if isinstance(q_max, tuple): # for copy
            self._xs, self._ys, self._ws = q_max
        else:
            self._xs = TensorQueue(q_max, data.shape[1])
            self._ys = TensorQueue(q_max, data.shape[0])
            self._ws = TensorQueue(q_max)

    def _update(self, x:_TS, y:_TS): #TODO: weight?
        #Batch Size! Batch Size! Batch Size!
        for x_n, y_n in zip(x, y):
            for idx, x_r in enumerate(self._xs.data):
                # print(x_n.shape, x_r.shape)
                # input()
                if 1 - F.normalize(torch.dot(x_r, x_n), dim=0) < self.eps:
                    self._ys[idx] += y_n
                    break
                #TODO: better weight
            else: # 没有相似的
                self._xs.push(x_n, True)
                self._ys.push(y_n, True)
                self._ws.push(1, True)
                #TODO: better drop
        '''#@hzw: 
        如果BatchSize过大，就会被用各种方式截掉。相当于用少量数据集进行更新。这个可以用近似者加权来干掉。
        权重确定好了也许很matter
        '''
    @property
    def grad(self):
        # x = $(QL, in_dim), y=$(QL, out_dim)
        # grad = y.T @ x
        return (self._ws.data * self._ys.data.T) @ self._xs.data
    @grad.setter
    def grad(self, v): pass
    
    def to(self, *args, **kwargs):
        return QueueParam(
            q_max=(
                self._xs.to(*args, **kwargs),
                self._ys.to(*args, **kwargs),
                self._ws.to(*args, **kwargs)
            ),
            data=self.data.to(*args, **kwargs),
            requires_grad=self.requires_grad
        )

class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A:_TS, x:_TS, bias:_TS = None):
        assert isinstance(A, QueueParam), f"Expect an instance of {QueueParam.__name__} but got {type(A).__name__}"
        ctx.save_for_backward(A, x, torch.tensor(bias is None, dtype=torch.bool))
        return nn.functional.linear(x, A, bias)

    @staticmethod
    def backward(ctx: Any, grad_output: _TS):
        A, x, bias_is_None = ctx.saved_tensors
        assert isinstance(A, QueueParam), f"Expect an instance of {QueueParam.__name__} but got {type(A).__name__}"
        grad_x = nn.functional.linear(grad_output, A.T)
        A._update(x, grad_output)
        # grad_A = A.grad
        if bias_is_None:
            return None, grad_x
        else:
            return None, grad_x, grad_output

class Linear(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, q_max = 64, bias = True):
        super().__init__()
        self.weight = QueueParam(
            q_max=q_max, 
            data=torch.ones((out_dim, in_dim), dtype=torch.float32),
            requires_grad=True
        )
        self.bias = torch.rand((out_dim,), requires_grad=True) if bias else None
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return LinearFunction.apply(self.weight, x, self.bias)

def check_in(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f'func <{func.__name__}> in')
        res = func(*args, **kwargs)
        print(f'func <{func.__name__}> out')
        return res
    return wrapper


# --------------------test main-----------------------


@check_in
def _test_layer():
    B = 3
    A = nn.Parameter(data=torch.tensor([
        [1., 1., 1.],
        [1., 1., 1.]
    ], requires_grad=True))
    b = torch.rand((2,), requires_grad=True)
    net = Linear(3, 2)
    x = torch.tensor(B*[[3., 4., 5.]])

    y1 = 20*nn.functional.linear(x, A, b).mean()
    y1.backward()
    print(A.grad, b.grad, end='\n----\n')
    A.grad = None

    yn = 20*net(x).mean()
    yn.backward()
    print(net.weight.grad, net.bias.grad, end='\n----\n')
    net.zero_grad()

    
if __name__=='__main__':
    # _test_queue()
    _test_layer()