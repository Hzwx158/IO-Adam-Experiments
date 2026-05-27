import torch


class GradBuffer:
    def __init__(self, in_dim: int, out_dim:int, beta1: float = None, beta2: float = None, p: float = 2, max_queue_length: int = 16, w_momentum: bool = True, clear_final: bool = False, merge_final: bool = False) -> None:
        self.full = False
        self.w_momentum = w_momentum
        if self.w_momentum:
            self.exp_avg = torch.zeros((out_dim, in_dim))
            self.beta1 = beta1
        else:
            self.input = None
            self.grad_output = None
        self.beta2 = beta2
        self.p = p
        # self.q = 1 / (1 - 1/ p)
        self.q = p / (p - 1)
        print(f"p={self.p}, q={self.q}")
        self.v_in_queue = torch.zeros(max_queue_length, in_dim)
        self.v_out_queue = torch.zeros(max_queue_length, out_dim)
        self.max_queue_length = max_queue_length
        self.pointer = 0
        self.clear_final = clear_final
        self.merge_final = merge_final
        #self.exp_avg_sq = torch.zeros((out_dim, in_dim))
        #self.count = 0


    @torch.no_grad()
    def update(self, input, grad_output):
        if self.w_momentum:
            if input.device != self.exp_avg.device:
                self.exp_avg = self.exp_avg.to(input.device)
            #self.exp_avg_sq = self.exp_avg_sq.to(self.exp_avg.device)
            self.exp_avg.add_(grad_output.t().mm(input), alpha=1-self.beta1)
        else:
            self.input = input
            self.grad_output = grad_output
        if input.device != self.v_out_queue.device:
            self.v_in_queue = self.v_in_queue.to(input.device)
            self.v_out_queue = self.v_out_queue.to(input.device)

        self.v_out_queue[self.pointer].add_(grad_output.abs().pow_(self.q).sum(dim=0), alpha=1-self.beta2)
        self.v_in_queue[self.pointer].add_(input.abs().pow_(self.p).sum(dim=0), alpha=1-self.beta2)


        #self.count += input.size(0)
        #self.exp_avg_sq.add_(grad_output.t().mm(input).pow(2), alpha=1-self.beta2)

    @torch.no_grad()
    def avg_decay(self):
        if self.w_momentum:
            self.exp_avg.mul_(self.beta1)
        self.v_out_queue.mul_(self.beta2)
        self.v_in_queue.mul_(self.beta2)

    def to(self, *args, **kwargs):
        self.exp_avg = self.exp_avg.to(*args, **kwargs)
        #self.exp_avg_sq = self.exp_avg_sq.to(*args, **kwargs)
        self.input_sq = self.input_sq.to(*args, **kwargs)
        self.grad_output_sq = self.grad_output_sq.to(*args, **kwargs)

    @property
    @torch.no_grad()
    def data(self):
        if self.pointer == self.max_queue_length - 1:
            self.full = True
        self.pointer = (self.pointer + 1) % self.max_queue_length
        if self.clear_final:
            if self.full:
                if self.merge_final:
                    next_pointer = (self.pointer + 1) % self.max_queue_length
                    self.v_out_queue[next_pointer].mul_(self.beta2).add_(self.v_out_queue[self.pointer],
                                                                         alpha=1 - self.beta2)
                    self.v_in_queue[next_pointer].mul_(self.beta2).add_(self.v_in_queue[self.pointer],
                                                                        alpha=1 - self.beta2)
                self.v_in_queue[self.pointer].mul_(0.)
                self.v_out_queue[self.pointer].mul_(0.)
        #return self.exp_avg, self.exp_avg_sq
        #return self.exp_avg, self.grad_output_sq, self.input_sq
        denom = torch.mm(
            self.v_out_queue.pow(1 / self.q).t(),
            self.v_in_queue.pow(1 / self.p)
        )
        if self.w_momentum:
            # return self.exp_avg, self.v_out_queue.t().mm(self.v_in_queue).sqrt()
            return self.exp_avg, denom
        else:
            # return self.grad_output.t().mm(self.input), self.v_out_queue.t().mm(self.v_in_queue).sqrt()
            return self.grad_output.t().mm(self.input), denom