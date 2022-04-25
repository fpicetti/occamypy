import torch

from occamypy.torch.vector import VectorTorch

__all__ = [
    "AutogradFunction",
    "VectorAD",
]


class VectorAD(VectorTorch):
    """
    VectorTorch child which allows tensors to be atteched to the graph (requires_grad=True)

    Notes:
        tensors are stored in C-contiguous memory
    """
    
    def __init__(self, in_content, device: int = None, *args, **kwargs):
        """
        VectorAD constructor

        Args:
            in_content: Vector, np.ndarray, torch.Tensor or tuple
            device: computation device (None for CPU, -1 for least used GPU)
            *args: list of arguments for Vector construction
            **kwargs: dict of arguments for Vector construction
        """
        if isinstance(in_content, VectorTorch):
            super(VectorAD, self).__init__(in_content=in_content[:], *args, **kwargs)
            self.getNdArray().requires_grad = True
        else:
            super(VectorAD, self).__init__(in_content=in_content, device=device, *args, **kwargs)
            if not self.getNdArray().requires_grad:
                self.getNdArray().requires_grad = True
    
    @property
    def requires_grad(self):
        return self.getNdArray().requires_grad
    
    @property
    def grad(self):
        return self.getNdArray().grad
    
    def backward(self, *args, **kwargs):
        return self.getNdArray().backward(*args, **kwargs)
    
    def max(self):
        max = self.getNdArray().max()
        return max
    
    def min(self):
        min = self.getNdArray().min()
        return min
    
    def norm(self, N=2):
        norm = torch.linalg.norm(self.getNdArray().flatten(), ord=N)
        return norm
    
    def zero(self):
        self[:] = torch.zeros_like(self.getNdArray())
        return self
    
    def set(self, val: float or int):
        self[:] = val * torch.ones_like(self[:])
        return self


class _Function(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, model: torch.Tensor, vec_class, fwd_fn, adj_fn, op_dev: torch.device = None,
                torch_comp: bool = True) -> torch.Tensor:
        ctx.fwd = fwd_fn
        ctx.adj = adj_fn
        ctx.op_dev = op_dev
        ctx.torch_comp = torch_comp
        ctx.vec_class = vec_class
        
        if ctx.torch_comp:
            # operator is torch-compatible, so we move the domain to the same device of the operator
            data = ctx.fwd(ctx.vec_class(model.clone().to(ctx.op_dev)))[:].to(model.device)
        else:
            # operator is not torch-compatible, so we move the domain to CPU to the same vec_class of the operator
            data = ctx.fwd(ctx.vec_class(model.detach().numpy()))
            data = torch.from_numpy(data[:]).to(model.device)
        return data
    
    @staticmethod
    def backward(ctx, data: torch.Tensor):
        
        if ctx.torch_comp:
            # operator is torch-compatible, so we move the domain to the same device of the operator
            model = ctx.adj(ctx.vec_class(data.to(ctx.op_dev)))[:].to(data.device)
        else:
            # operator is not torch-compatible, so we move the domain to CPU
            model = ctx.adj(data.detach().numpy())
            model = torch.from_numpy(model[:]).to(data.device)
        return model, None, None, None, None, None


class AutogradFunction:
    """
    Cast a Operator to a Autograd Function to be used in the torch graph

    Examples:
        T = ToAutograd(my_operator)

        T(tensor) -> tensor

        T(vector) -> tensor

        T * vector -> vector
    """
    
    def __init__(self, operator):
        """
        AutogradFunction constructor

        Args:
            operator: linear operator, can be based on any vector backend
        """
        # check if operator is torch compatible
        _is_vectorAD = isinstance(operator.domain, VectorAD) and isinstance(operator.range, VectorAD)
        _is_vectorTorch = isinstance(operator.domain, VectorTorch) and isinstance(operator.range, VectorTorch)
        self.vec_class = operator.range.__class__
        
        if _is_vectorAD or _is_vectorTorch:
            self.torch_comp = True
        else:
            self.torch_comp = False
        
        # check the device on which the operator runs
        if self.torch_comp:
            try:
                self.op_dev = operator.device
            except AttributeError:
                self.op_dev = operator.domain.device
        else:  # is not a torch based operator
            self.op_dev = "cpu"
        
        self.fwd = lambda x: operator * x
        self.adj = lambda x: operator.T * self.vec_class(x)
        self.function = _Function.apply
    
    def __mul__(self, other):  # occamypy notation: self * vector -> vector
        out = self(other)
        return other.__class__(out)
    
    def __call__(self, other):  # torch notation: self(tensor, vector) -> tensor
        if isinstance(other, torch.Tensor):
            return self.apply(other)
        else:
            return self.apply(other.getNdArray())
    
    def apply(self, model: torch.Tensor) -> torch.Tensor:
        return self.function(model, self.vec_class,
                             self.fwd, self.adj,
                             self.op_dev, self.torch_comp, )


if __name__ == "__main__":
    import occamypy as o
    
    S = torch.nn.Sigmoid()
    
    # use with VectorTorch (requires_grad=False)
    x = o.VectorTorch(torch.ones(2))
    T = AutogradFunction(o.Scaling(x, 2))
    y_ = T(x)
    sig_y_ = S(T(x))
    y_sig = T(S(x[:]))
    y = T * x
    del y, y_, sig_y_, y_sig
    
    # use with VectorAD (requires_grad=True) to wrap learnable tensors
    x = VectorAD(torch.ones(2))
    T = AutogradFunction(o.Scaling(x, 2))
    y_ = T(x)
    sig_y_ = S(T(x))
    y_sig = T(S(x[:]))
    y = T * x
    del y, y_, sig_y_, y_sig
    
    # now try a numpy-based operator on a tensor with requires_grad=False
    x = VectorTorch(torch.zeros(21))
    x[10] = 1
    T = AutogradFunction(o.GaussianFilter(o.VectorNumpy((x.size,)), 1))
    y_ = T(x)
    sig_y_ = S(T(x))
    y_sig = T(S(x[:]))
    y = T * x
    del y, y_, sig_y_, y_sig
    
    # now try a numpy-based operator on a tensor with requires_grad=True
    x = VectorAD(x)
    T = AutogradFunction(o.GaussianFilter(o.VectorNumpy((x.size,)), 1))
    y_ = T(x)
    sig_y_ = S(T(x))
    y_sig = T(S(x[:]))
    y = T * x
    
    print(0)
