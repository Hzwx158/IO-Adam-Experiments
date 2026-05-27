import torch
from torch import nn
from torch.optim import Optimizer, AdamW
from typing import Any, Iterable, Callable, Iterable, Tuple, Dict, List
from functools import wraps
import math
import warnings
from torch import Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter, UninitializedParameter

from transformers.utils.versions import require_version
from .grad_buffer import GradBuffer
# from .grad_buffer_try_best import GradBuffer
from .decomposed_linear import DecomposedLinear
import gc
from tqdm import tqdm


class AdamW_Decomposed(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
            self,
            params: Iterable[nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            p:float = 1.1,
            max_queue_length:int = 16,
            w_momentum:bool = True,
            clear_final:bool = False,
            merge_final:bool = False
    ):
        require_version("torch>=1.5.0")  # add_ with alpha
        print(lr, max_queue_length)
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        self.p = p
        self.max_queue_length = max_queue_length
        self.w_momentum = w_momentum
        self.clear_final = clear_final
        self.merge_final = merge_final
        super().__init__(params, defaults)

    def __param_to_module__(self, model) -> Dict[Parameter, Tuple[nn.Linear, str]]:
        # get the mapping from parameters to all Linear Module
        param_to_module = {}
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                for name, param in module.named_parameters(recurse=False):  # 只看当前 module 的参数
                    full_name = f"{module_name}.{name}" if module_name else name
                    if "weight" in full_name:
                        param_to_module[param] = (module, module_name)
                    else:
                        param_to_module[param] = (None, None)
        return param_to_module

    def replace_module(self, model):
        param_to_module = self.__param_to_module__(model)
        for group in self.param_groups:
            new_params = []
            replaced = set()
            beta1, beta2 = group["betas"]
            for p in group['params']:
                p:Parameter
                if p in param_to_module.keys():
                    module, module_name = param_to_module[p]
                    if module is not None and id(module) not in replaced:
                        if p in self.state.keys():
                            state = self.state.pop(p)
                        else:
                            state = dict()
                        new_module = DecomposedLinear(
                            module.in_features, 
                            module.out_features, 
                            bias=(module.bias is not None)
                        ).to(p.device)
                        new_module.weight.data.copy_(module.weight.data)
                        if module.bias is not None:
                            new_module.bias.data.copy_(module.bias.data)
                        grad_buffer = GradBuffer(module.in_features, module.out_features, beta1, beta2, self.p, self.max_queue_length, self.w_momentum, self.clear_final, self.merge_final)
                        state["grad_buffer"] = grad_buffer
                        new_module.prepare_decompose_linear(grad_buffer)

                        parent = model
                        if '.' in module_name:
                            path = module_name.split('.')
                            for p in path[:-1]:
                                parent = getattr(parent, p)
                            setattr(parent, path[-1], new_module)
                        else:
                            setattr(model, module_name, new_module)
                        new_params += [p for p in new_module.parameters() if p.requires_grad]
                        self.state[new_module.weight] = state
                        replaced.add(id(module))
                    else:
                        pass
                else:
                    new_params.append(p)
            group["params"] = new_params


    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                if p.grad is None:
                    state = self.state[p]
                    if "grad_buffer" in state.keys():
                        if "step" not in state:
                            state["step"] = 0
                        state["step"] += 1
                        
                        # if state["step"] == 2:
                        #     tqdm.write(str(torch.cuda.memory_allocated(p.device) / 1024**3))
                        # exp_avg, denom = state["grad_buffer"].data(group["eps"])
                        exp_avg, denom = state["grad_buffer"].data
                        denom.add_(group["eps"])
                        step_size = group["lr"]
                        if group["correct_bias"]:  # No bias correction for Bert
                            bias_correction1 = 1.0 - beta1 ** state["step"]
                            bias_correction2 = 1.0 - beta2 ** state["step"]
                            step_size = step_size * bias_correction2 / bias_correction1
                        p.addcdiv_(exp_avg, denom, value=-step_size)
                        # if state["step"] == 2:
                        #     tqdm.write(str(torch.cuda.memory_allocated(p.device) / 1024**3))
                        # exp_avg = denom = exp_avg_sq = None
                        del exp_avg, denom
                        # if state["step"] == 2:
                        #     tqdm.write(str(torch.cuda.memory_allocated(p.device) / 1024**3))
                        #p.add_(exp_avg, alpha=-step_size)
                        state["grad_buffer"].avg_decay()
                    else:
                        continue
                else:
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                    state = self.state[p]

                    if "step" not in state:
                        state["step"] = 0

                    # State initialization
                    if "exp_avg" not in state:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]

                    state["step"] += 1

                    # Decay the first and second moment running average coefficient
                    # In-place operations to update the averages at the same time
                    exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                    step_size = group["lr"]
                    if group["correct_bias"]:  # No bias correction for Bert
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size = step_size * bias_correction2 / bias_correction1

                    # compute norm gradient
                    norm_grad = exp_avg / denom

                    p.add_(norm_grad, alpha=-step_size)



        return loss

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    model = AutoModelForCausalLM.from_pretrained("/home_new/chenyiting/RoPE_angle/model/Llama-2-7b-hf",torch_dtype=torch.float16, trust_remote_code=True)
    optimizer = AdamW_Decomposed(model.parameters(), lr=1e-3)
    import time
    tick = time.time()
    optimizer.replace_module(model)
    tock = time.time()
    print(tock - tick)
    optimizer.test_queue()
    '''for name, module in model.named_modules():
        if isinstance(module, DecomposedLinear):
            print(name)'''

