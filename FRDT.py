"""Future-Aware Resource Dependency Transformer (FRDT).

Triangular O2O, RoPE, duration edge attention and machine self-attention follow
ReSched. FRDT adds explicit future load and action-conditioned delay propagation;
triangular attention alone is not a separate novelty claim.
"""
import math
import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-8) * self.scale + self.bias


def split_heads(x, heads):
    x = x.reshape(*x.shape[:-1], heads, x.shape[-1] // heads)
    return x.movedim(-2, 1)


def rope(q, k, position):
    dim = q.shape[-1]
    frequency = torch.exp(-math.log(10000.) * torch.arange(0, dim, 2, device=q.device) / dim)
    angle = position[..., None] * frequency
    c = angle.cos().repeat_interleave(2, -1)[:, None]
    s = angle.sin().repeat_interleave(2, -1)[:, None]
    def rotate(x):
        return torch.stack((-x[..., 1::2], x[..., ::2]), -1).flatten(-2)
    return q * c + rotate(q) * s, k * c + rotate(k) * s


def masked_softmax(scores, allowed):
    allowed = allowed[:, None]
    scores = scores.masked_fill(~allowed, -torch.inf)
    scores = torch.where(allowed.any(-1, keepdim=True), scores, torch.zeros_like(scores))
    return scores.softmax(-1).masked_fill(~allowed, 0)


class Attention(nn.Module):
    def __init__(self, dim, heads, ff_dim):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dim))

    def forward(self, query, source, mask, position=None, edge=None, self_channel=False):
        q, k, v = [split_heads(x, self.heads) for x in (self.q(query), self.k(source), self.v(source))]
        if position is not None:
            q, k = rope(q, k, position)
        score = q @ k.transpose(-2, -1)
        if edge is not None:
            e = split_heads(edge, self.heads)
            score = (score + torch.einsum('bhmd,bhmod->bhmo', q, e)
                     + torch.einsum('bhod,bhmod->bhmo', k, e) + e.square().sum(-1))
        allowed = mask
        if self_channel:
            score = torch.cat((score, q.square().sum(-1, keepdim=True)), -1)
            allowed = torch.cat((mask, torch.ones_like(mask[..., :1])), -1)
        weights = masked_softmax(score * self.scale, allowed)
        w = weights[..., :-1] if self_channel else weights
        context = w @ v
        if edge is not None:
            context = context + torch.einsum('bhmo,bhmod->bhmd', w, e)
        if self_channel:
            context = context + weights[..., -1:] * q
        context = context.transpose(1, 2).flatten(-2)
        out = self.norm1(query + self.out(context))
        out = self.norm2(out + self.ff(out))
        return torch.where(mask.any(-1, keepdim=True), out, query)


class CrossBlock(nn.Module):
    def __init__(self, dim, heads, ff_dim):
        super().__init__()
        self.operation = Attention(dim, heads, ff_dim)
        self.machine = Attention(dim, heads, ff_dim)

    def forward(self, operation, machine, o_mask, m_mask, edge, position):
        o = self.operation(operation, operation, o_mask, position=position)
        m = self.machine(machine, operation, m_mask.transpose(1, 2),
                         edge=edge.transpose(1, 2), self_channel=True)
        return o, m


def future_resource_demand(obs):
    """Future p[o,m]/degree[o], summed by job; excludes current candidates.

    Durations are already divided by the instance's fixed maximum duration.
    This is expected load under uniform assignment, not a reservation/forecast.
    Machine-specific times preserve SD2 heterogeneity. log1p limits scale growth.
    """
    duration, job = obs['duration'], obs['job']
    degree = (duration > 0).sum(-1).clamp_min(1)
    future = obs['valid'] & ~obs['pending']
    contribution = duration / degree[..., None] * future[..., None]
    # Pool by job without allocating another quadratic operation-operation tensor.
    job_count = int(job.max().item()) + 1
    job_load = duration.new_zeros(duration.shape[0], job_count, duration.shape[2])
    index = job[..., None].expand_as(duration)
    job_load.scatter_add_(1, index, contribution)
    candidate_load = job_load.gather(1, index) * obs['pending'][..., None]
    return torch.log1p(candidate_load), torch.log1p(contribution.sum(1))


@torch.no_grad()
def compact_state_features(obs, successor_mask):
    """Four operation and three machine descriptors from the current state.

    These supplement minimum duration/job readiness and machine availability.
    Counts and suffix work expose information that normalized attention need
    not retain. Durations use the environment's fixed per-instance scale;
    counts/work use log1p. No feature depends on batch padding or future actions.
    """
    duration, valid = obs['duration'], obs['valid']
    compatible = (duration > 0) & valid[..., None]
    degree = compatible.sum(-1).to(duration.dtype)
    connected = compatible.any(1)
    machines = connected.sum(-1, keepdim=True).clamp_min(1)
    suffix_count = successor_mask.sum(-1).to(duration.dtype)
    suffix_work = torch.bmm(successor_mask.to(duration.dtype),
                            (obs['minimum'] * valid)[..., None]).squeeze(-1)
    operation = torch.stack((duration.sum(-1) / degree.clamp_min(1),
                             degree / machines,
                             torch.log1p(suffix_work),
                             torch.log1p(suffix_count)), -1)
    remaining_count = compatible.sum(1).to(duration.dtype)
    candidate_count = (compatible & obs['pending'][..., None]).sum(1).to(duration.dtype)
    machine = torch.stack((duration.sum(1) / remaining_count.clamp_min(1),
                           torch.log1p(candidate_count),
                           torch.log1p(remaining_count)), -1)
    return operation * valid[..., None], machine * connected[..., None]


@torch.no_grad()
def future_delay_propagation(obs, successor_mask):
    """Two candidate-edge signals from relaxed completion-time propagation.

    For each remaining operation o, let p_o be its minimum duration and r_o
    the earliest availability of an eligible machine. The relaxed job chain
    follows E_o = max(E_predecessor, r_o) + p_o. It ignores future contention
    and may use different machines to attain r_o and p_o, so it is optimistic,
    not a simulated schedule or a prediction of actual completion times.

    For action (j,m), use its actual finish time c=max(job_at_j,m_at_m)+p_jm.
    Signal 1 is j's propagated completion increase over its best current
    machine choice, after downstream waiting has absorbed any delay.
    Signal 2 is the mean increase of other unfinished jobs' relaxed completion
    times when machine m is reserved until c. Each other job is counted once.
    The max-plus chain is evaluated by suffix work and release barriers;
    no environment copies, rollouts, or additional optimization are required.
    """
    duration, job = obs['duration'], obs['job']
    valid, pending = obs['valid'], obs['pending']
    compatible = (duration > 0) & valid[..., None]
    batch, operations, machines = duration.shape
    jobs = int(job.max().item()) + 1
    minimum = obs['minimum'] * valid
    # successor_mask includes the operation itself and its remaining successors.
    suffix_work = torch.bmm(successor_mask.to(duration.dtype),
                            minimum[..., None]).squeeze(-1)
    releases = obs['m_at'][:, None, :].expand_as(duration).masked_fill(
        ~compatible, float('inf'))
    first_release, first_machine = releases.min(-1)
    first_release = first_release.masked_fill(~valid, 0)
    if machines > 1:
        second_release = releases.topk(2, dim=-1, largest=False).values[..., 1]
    else:
        second_release = torch.full_like(first_release, float('inf'))
    machine_ids = torch.arange(machines, device=duration.device)
    alternative_release = torch.where(
        first_machine[..., None] == machine_ids,
        second_release[..., None], first_release[..., None])

    job_ready = duration.new_zeros(batch, jobs).scatter_add_(
        1, job, obs['o_at'] * pending)
    work = duration.new_zeros(batch, jobs).scatter_add_(1, job, minimum)
    current_minimum = duration.new_zeros(batch, jobs).scatter_add_(
        1, job, minimum * pending)
    current_duration = duration.new_zeros(batch, jobs, machines).scatter_add_(
        1, job[..., None].expand_as(duration), duration * pending[..., None])
    eligible = current_duration > 0
    active = eligible.any(-1)
    finish = torch.maximum(job_ready[..., None], obs['m_at'][:, None, :]) + current_duration
    barrier = (first_release + suffix_work).masked_fill(~valid, 0)
    baseline = (job_ready + work).scatter_reduce(
        1, job, barrier, reduce='amax', include_self=True)
    future_barrier = duration.new_zeros(batch, jobs).scatter_reduce_(
        1, job, barrier.masked_fill(~(valid & ~pending), 0),
        reduce='amax', include_self=True)

    own_finish = torch.maximum(finish + (work - current_minimum)[..., None],
                               future_barrier[..., None])
    best_own_finish = own_finish.masked_fill(~eligible, float('inf')).amin(-1)
    best_own_finish = best_own_finish.masked_fill(~active, 0)
    own_delay = (own_finish - best_own_finish[..., None]).clamp_min(0)

    # Raising only m's availability changes an operation's release to
    # min(c, earliest_alternative) if m was its earliest machine. Otherwise
    # it cannot raise the release. Baseline already includes all old barriers.
    changed_barrier = (torch.minimum(finish[:, :, None, :],
                                     alternative_release[:, None, :, :])
                       + suffix_work[:, None, :, None])
    operation_delay = (changed_barrier
                       - baseline.gather(1, job)[:, None, :, None]).clamp_min(0)
    other_job = job[:, None, :] != torch.arange(jobs, device=job.device)[None, :, None]
    operation_delay.masked_fill_(~(valid[:, None, :] & other_job)[..., None], 0)
    job_delay = duration.new_zeros(batch, jobs, jobs, machines).scatter_reduce_(
        2, job[:, None, :, None].expand_as(operation_delay), operation_delay,
        reduce='amax', include_self=True)
    other_count = (active.sum(-1) - 1).clamp_min(1).to(duration.dtype)
    external_delay = job_delay.sum(2) / other_count[:, None, None]

    delay = torch.stack((own_delay, external_delay), -1)
    delay.masked_fill_(~eligible[..., None], 0)
    delay = delay.gather(1, job[..., None, None].expand(batch, operations, machines, 2))
    delay = delay.masked_fill(~obs['eligible'][..., None], 0)
    return torch.log1p(delay)


class FRDT(nn.Module):
    def __init__(self, embedding_dim=128, head_num=8, block_num=2,
                 ff_hidden_dim=512, use_future_demand=True, use_delay_propagation=True):
        super().__init__()
        if min(embedding_dim, head_num, block_num, ff_hidden_dim) < 1:
            raise ValueError('FRDT dimensions, heads and block count must be positive.')
        if embedding_dim % head_num or (embedding_dim // head_num) % 2:
            raise ValueError('embedding_dim/head_num must be an even integer for RoPE.')
        self.use_future_demand = use_future_demand
        self.use_delay_propagation = use_delay_propagation
        # One independent projection per complete feature vector: 6 / 5 / 4 -> d.
        self.operation_embedding = nn.Linear(6, embedding_dim)
        self.machine_embedding = nn.Linear(5, embedding_dim)
        self.pair_embedding = nn.Linear(4, embedding_dim)
        # Retain weak initial supplementary/load weights and neutral delay weights.
        nn.init.normal_(self.operation_embedding.weight[:, 2:], std=0.01)
        nn.init.normal_(self.machine_embedding.weight[:, 1:], std=0.01)
        nn.init.normal_(self.pair_embedding.weight[:, 1:2], std=0.01)
        nn.init.zeros_(self.pair_embedding.weight[:, 2:])
        self.layers = nn.ModuleList([CrossBlock(embedding_dim, head_num, ff_hidden_dim) for _ in range(block_num)])

    def forward(self, obs):
        valid = obs['valid']
        position, job = obs['position'], obs['job']
        # Positions count down to zero: >= selects the query itself and its successors.
        o_mask = ((job[:, :, None] == job[:, None, :])
                  & (position[:, :, None] >= position[:, None, :])
                  & valid[:, :, None] & valid[:, None, :])
        operation_extra, machine_extra = compact_state_features(obs, o_mask)
        if self.use_future_demand:
            pair_load, machine_load = future_resource_demand(obs)
        else:
            pair_load = torch.zeros_like(obs['duration'])
            machine_load = torch.zeros_like(obs['m_at'])
        if self.use_future_demand and self.use_delay_propagation:
            delay = future_delay_propagation(obs, o_mask)
        else:
            delay = obs['duration'].new_zeros(*obs['duration'].shape, 2)

        # Supplementary descriptors are always included. The environment already
        # gates o_at to candidates; future-load and delay gating is unchanged.
        operation_features = torch.cat((obs['minimum'][..., None],
                                        obs['o_at'][..., None], operation_extra), -1)
        machine_features = torch.cat((obs['m_at'][..., None], machine_extra,
                                      machine_load[..., None]), -1)
        pair_features = torch.cat((obs['duration'][..., None],
                                   pair_load[..., None], delay), -1)
        operation = self.operation_embedding(operation_features) * valid[..., None]
        machine = self.machine_embedding(machine_features)
        pair_edge = self.pair_embedding(pair_features)
        m_mask = (obs['duration'] > 0) & valid[..., None]
        # Send the explicit downstream demand to machine attention as well as the actor.
        for layer in self.layers:
            operation, machine = layer(operation, machine, o_mask, m_mask, pair_edge, position)
            operation = operation * valid[..., None]
        return operation, machine, pair_edge
