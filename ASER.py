"""Adaptive Structured Entropy Regularization (ASER) for FAGT.

The policy minimizes alpha * normalized KL(pi || q), where q allocates equal
mass to unfinished jobs and then to their eligible machines. The adaptive
coefficient acts on the exploration term of the PPO objective.
"""
import math
import torch


def job_balanced_reference(eligible, dtype=torch.float32):
    """One nonempty candidate row per unfinished job; zero on invalid pairs."""
    degree = eligible.sum(-1).to(dtype)
    jobs = (degree > 0).sum(-1).to(dtype)
    denominator = jobs[:, None, None].clamp_min(1) * degree[..., None].clamp_min(1)
    return (eligible.to(dtype) / denominator).flatten(1)


def structural_kl(prob, eligible):
    """KL(pi || q) / log(J * maximum_degree); forced states contribute zero."""
    q = job_balanced_reference(eligible, prob.dtype)
    tiny = torch.finfo(prob.dtype).tiny
    log_ratio = prob.clamp_min(tiny).log() - q.clamp_min(tiny).log()
    kl = (prob * log_ratio).sum(-1).clamp_min(0)
    degree = eligible.sum(-1).to(prob.dtype)
    maximum = ((degree > 0).sum(-1).to(prob.dtype).clamp_min(1).log()
               + degree.amax(-1).clamp_min(1).log())
    choice = eligible.flatten(1).sum(-1) > 1
    statistic = torch.where(choice, kl / maximum.clamp_min(1e-8), torch.zeros_like(kl))
    return statistic, choice


class ASER:
    def __init__(self, coefficient=0.01, target=0.5, lr=3e-4,
                 minimum=1e-4, maximum=0.1, device='cpu'):
        if not 0 < minimum <= coefficient <= maximum or not 0 <= target <= 1 or lr <= 0:
            raise ValueError('ASER requires 0 < minimum <= coefficient <= maximum, target in [0,1], lr > 0.')
        self.target, self.minimum, self.maximum = float(target), float(minimum), float(maximum)
        self.log_alpha = torch.tensor(math.log(coefficient), device=device, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def coefficient(self):
        return self.log_alpha.exp().detach()

    def penalty(self, prob, eligible):
        statistic, choice = structural_kl(prob, eligible)
        return self.coefficient * statistic.mean(), statistic, choice

    def update(self, statistic):
        """One detached controller update per rollout; caller excludes forced states."""
        statistic = torch.as_tensor(statistic, device=self.log_alpha.device).detach()
        if statistic.numel() != 1 or not torch.isfinite(statistic).all():
            raise ValueError('ASER statistic must be a finite scalar.')
        # Above the KL budget, gradient descent increases the penalty coefficient.
        loss = self.log_alpha * (self.target - statistic)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            self.log_alpha.clamp_(math.log(self.minimum), math.log(self.maximum))

