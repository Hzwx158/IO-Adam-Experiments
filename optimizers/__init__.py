from .Adamw_w_grad_decompose import AdamW_Decomposed
from .Adamw_w_grad_decompose_v2 import AdamW_Decomposed as AdamW_Decomposed_v2
from torch import optim

decomposed_types = [
    AdamW_Decomposed,
    AdamW_Decomposed_v2
]

def make_optimizer(optimizer_choice:str, model, *args, **kwargs)->optim.Optimizer:
    optimizer = eval(optimizer_choice)(*args, **kwargs)
    if any(isinstance(optimizer, T) for T in decomposed_types):
        optimizer.replace_module(model = model)
        print('replace_module over')
    return optimizer
    
