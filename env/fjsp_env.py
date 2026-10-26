"""Append-scheduling FJSP environment with ReSched-style remaining subproblems.

No clock-based pruning: every unfinished job's next operation can be reserved
on any compatible machine. Start = max(job availability, machine availability).
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch


@dataclass
class Instance:
    lengths: np.ndarray
    duration: np.ndarray
    name: str = ''

    def __post_init__(self):
        lengths = np.asarray(self.lengths)
        if lengths.ndim != 1 or not np.isfinite(lengths).all() or not np.equal(lengths, lengths.astype(np.int64)).all():
            raise ValueError('Job lengths must be a vector of integers.')
        self.lengths = lengths.astype(np.int64)
        self.duration = np.asarray(self.duration, dtype=np.float32)
        if (self.lengths <= 0).any() or self.duration.ndim != 2 or self.duration.shape[1] == 0:
            raise ValueError('Every job and operation must be nonempty.')
        if self.lengths.sum() != len(self.duration):
            raise ValueError('sum(lengths) must equal the number of operations.')
        if not np.isfinite(self.duration).all() or (self.duration < 0).any() or not (self.duration > 0).any(1).all():
            raise ValueError('Every operation needs a positive finite duration on at least one machine.')


def generate_instances(count, jobs=10, machines=5, distribution='SD2', seed=None,
                       rng=None, min_operations=None, max_operations=None,
                       min_processing_time=None, max_processing_time=None):
    """Generate SD1 (correlated operation times) or SD2 (independent times).

    The training path uses CaseGenerator_SD1/CaseGenerator directly. This helper
    uses a local NumPy RNG so callers can reproduce cases without global state.
    """
    rng = np.random.default_rng(seed) if rng is None else rng
    if count < 1 or jobs < 1 or machines < 1 or distribution not in ('SD1', 'SD2'):
        raise ValueError('Invalid generator dimensions or distribution.')
    lo = (max(1, int(0.8 * machines)) if distribution == 'SD1' else machines) if min_operations is None else min_operations
    hi = (max(1, int(1.2 * machines)) if distribution == 'SD1' else machines) if max_operations is None else max_operations
    low_pt = 1 if min_processing_time is None else min_processing_time
    high_pt = (20 if distribution == 'SD1' else 99) if max_processing_time is None else max_processing_time
    if not (1 <= lo <= hi and 0 < low_pt <= high_pt):
        raise ValueError('Invalid operation-count or processing-time bounds.')
    sd2_lengths = rng.integers(lo, hi + 1, size=jobs) if distribution == 'SD2' else None
    result = []
    for i in range(count):
        lengths = rng.integers(lo, hi + 1, size=jobs) if distribution == 'SD1' else sd2_lengths
        for attempt in range(10000):
            duration = np.zeros((int(lengths.sum()), machines), dtype=np.float32)
            degrees = rng.integers(1, machines + 1, len(duration))
            for o, degree in enumerate(degrees):
                selected = rng.choice(machines, degree, replace=False)
                if distribution == 'SD1':
                    mean = int(rng.integers(low_pt, high_pt + 1))
                    low = max(low_pt, round(mean * .8))
                    high = min(high_pt, round(mean * 1.2))
                    duration[o, selected] = rng.integers(low, high + 1, size=degree)
                else:
                    duration[o, selected] = rng.integers(low_pt, high_pt + 1, size=degree)
            if (duration > 0).any(0).all():
                break
        else:
            raise RuntimeError('Could not generate an instance covering all machines.')
        result.append(Instance(lengths.copy(), duration, f'{distribution}_{i}'))
    return result


def read_fjs(path):
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    jobs, machines = map(int, lines[0].split()[:2])
    if len(lines) < jobs + 1:
        raise ValueError(f'Incomplete FJS file: {path}')
    lengths, durations = [], []
    for line in lines[1:jobs + 1]:
        tokens = list(map(int, line.split()))
        n, k = tokens[0], 1
        lengths.append(n)
        for _ in range(n):
            degree = tokens[k]
            k += 1
            row = np.zeros(machines, dtype=np.float32)
            for _ in range(degree):
                machine, duration = tokens[k:k + 2]
                k += 2
                if not 1 <= machine <= machines or duration <= 0 or row[machine - 1] != 0:
                    raise ValueError(f'Invalid or repeated machine in {path}')
                row[machine - 1] = duration
            durations.append(row)
        if k != len(tokens):
            raise ValueError(f'Unexpected tokens in job line: {path}')
    return Instance(np.array(lengths), np.array(durations), Path(path).name)


def load_instances(path):
    """Read FJS/text files or numeric NPZ (native, DANIEL, or ReSched layout)."""
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in ('.fjs', '.txt', '.npz'))
        result = [inst for file in files for inst in load_instances(file)]
        if not result:
            raise ValueError(f'No .fjs/.txt/.npz instances in {path}')
        return result
    if path.suffix.lower() != '.npz':
        return [read_fjs(path)]
    with np.load(path, allow_pickle=False) as data:
        if 'arr_2' in data:  # ReSched: job ids, within-job positions, durations
            job_ids, positions, duration = data['arr_0'], data['arr_1'], data['arr_2']
            result = []
            for i in range(len(duration)):
                ids = np.unique(job_ids[i])
                order = np.lexsort((positions[i], job_ids[i]))
                lengths = np.array([(job_ids[i] == j).sum() for j in ids])
                result.append(Instance(lengths, duration[i, order], f'{path.stem}_{i}'))
            return result
        if 'job_lengths' in data:
            lengths = data['job_lengths']
            duration = data['processing_times'] if 'processing_times' in data else data['op_pt']
        else:
            lengths, duration = data['arr_0'], data['arr_1']
        if lengths.ndim == 1:
            lengths, duration = lengths[None], duration[None]
        return [Instance(l[l > 0], d[:int(l.sum())], f'{path.stem}_{i}')
                for i, (l, d) in enumerate(zip(lengths, duration))]


def instance_from_case_lines(lines, name=''):
    """Parse this framework's CaseGenerator output without writing a data file."""
    content = [line.strip() for line in lines if line.strip()]
    jobs, machines = map(int, content[0].split()[:2])
    if len(content) != jobs + 1:
        raise ValueError('CaseGenerator returned an incomplete FJS instance.')
    lengths, durations = [], []
    for line in content[1:]:
        values = list(map(int, line.split()))
        count, cursor = values[0], 1
        lengths.append(count)
        for _ in range(count):
            degree = values[cursor]
            cursor += 1
            row = np.zeros(machines, dtype=np.float32)
            for _ in range(degree):
                machine, processing_time = values[cursor:cursor + 2]
                cursor += 2
                if machine < 1 or machine > machines or processing_time <= 0 or row[machine - 1] != 0:
                    raise ValueError('Invalid generated operation-machine pair.')
                row[machine - 1] = processing_time
            durations.append(row)
        if cursor != len(values):
            raise ValueError('Invalid generated job line.')
    return Instance(np.array(lengths), np.array(durations), name)


def generate_case_instances(case, count):
    return [instance_from_case_lines(case.get_case(i)[0], f'case_{i}') for i in range(count)]


def save_instances(path, instances):
    max_j = max(len(x.lengths) for x in instances)
    max_o = max(len(x.duration) for x in instances)
    max_m = max(x.duration.shape[1] for x in instances)
    lengths = np.zeros((len(instances), max_j), dtype=np.int64)
    duration = np.zeros((len(instances), max_o, max_m), dtype=np.float32)
    for i, x in enumerate(instances):
        lengths[i, :len(x.lengths)] = x.lengths
        duration[i, :len(x.duration), :x.duration.shape[1]] = x.duration
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, job_lengths=lengths, processing_times=duration)


class FJSPEnv:
    def __init__(self, instances, device='cpu'):
        if not instances:
            raise ValueError('No instances supplied.')
        self.instances = instances
        self.device = torch.device(device)
        self.B = len(instances)
        self.O = max(len(x.duration) for x in instances)
        self.M = max(x.duration.shape[1] for x in instances)
        self.J = max(len(x.lengths) for x in instances)
        self.duration = torch.zeros(self.B, self.O, self.M, device=self.device)
        self.job = torch.zeros(self.B, self.O, dtype=torch.long, device=self.device)
        self.position = torch.zeros_like(self.job)
        self.valid = torch.zeros(self.B, self.O, dtype=torch.bool, device=self.device)
        self.first = torch.zeros(self.B, self.J, dtype=torch.long, device=self.device)
        self.end = torch.zeros_like(self.first)
        for b, instance in enumerate(instances):
            n, m = instance.duration.shape
            self.duration[b, :n, :m] = torch.as_tensor(instance.duration, device=self.device)
            self.valid[b, :n] = True
            start = 0
            for j, size in enumerate(instance.lengths):
                end = start + int(size)
                self.job[b, start:end] = j
                self.position[b, start:end] = torch.arange(size - 1, -1, -1, device=self.device)
                self.first[b, j], self.end[b, j] = start, end
                start = end
        self.scale = self.duration.amax((1, 2)).clamp_min(1)
        self.minimum = self.duration.masked_fill(self.duration == 0, float('inf')).amin(-1)
        self.minimum = self.minimum.masked_fill(~self.valid, 0)
        self.reset()

    def reset(self):
        self.next = self.first.clone()
        self.job_at = torch.zeros(self.B, self.J, device=self.device)
        self.machine_at = torch.zeros(self.B, self.M, device=self.device)
        self.remaining = self.valid.clone()
        self.remaining_work = torch.zeros_like(self.job_at).scatter_add_(1, self.job, self.minimum)
        self.lower_bound = self.remaining_work.amax(1)
        self.initial_lower_bound = self.lower_bound.clone()
        self.scheduled_machine = torch.full_like(self.job, -1)
        self.start = torch.zeros_like(self.minimum)
        self.finish = torch.zeros_like(self.minimum)
        self.done = ~self.remaining.any(1)
        return self.observe()

    @property
    def makespan(self):
        return self.finish.amax(1)

    def observe(self):
        """Remove finished batch rows and scheduled operation nodes, padding only
        unequal remaining lengths. One preprocessing path for all decoding modes.
        """
        batch = (~self.done).nonzero().squeeze(-1)
        if batch.numel() == 0:
            return None
        remaining = self.remaining[batch]
        n = int(remaining.sum(1).max())
        op_ids = torch.arange(self.O, device=self.device).expand(len(batch), -1)
        op_ids = op_ids.masked_fill(~remaining, self.O).sort(1).values[:, :n]
        valid = op_ids < self.O
        op_ids = op_ids.clamp_max(self.O - 1)
        b = batch[:, None]
        job = self.job[b, op_ids]
        duration = self.duration[b, op_ids] * valid[..., None]
        pending = (op_ids == self.next[batch].gather(1, job)) & valid
        eligible = (duration > 0) & pending[..., None]
        oat = self.job_at[batch].gather(1, job)
        mat = self.machine_at[batch]
        connected = (duration > 0).any(1)
        horizon = torch.minimum(oat.masked_fill(~pending, float('inf')).amin(1),
                                mat.masked_fill(~connected, float('inf')).amin(1))
        scale = self.scale[batch]
        obs = {
            'duration': duration / scale[:, None, None],
            'minimum': self.minimum[b, op_ids] * valid / scale[:, None],
            'o_at': ((oat - horizon[:, None]) / scale[:, None]) * pending,
            'm_at': ((mat - horizon[:, None]) / scale[:, None]) * connected,
            'job': job, 'position': self.position[b, op_ids],
            'valid': valid, 'pending': pending, 'eligible': eligible,
            'batch': batch, 'op_ids': op_ids,
        }
        return obs

    def step(self, actions, obs):
        b = obs['batch']
        seq = torch.div(actions, self.M, rounding_mode='floor')
        machine = actions % self.M
        row = torch.arange(len(b), device=self.device)
        if not obs['eligible'][row, seq, machine].all():
            raise ValueError('Ineligible action.')
        operation = obs['op_ids'][row, seq]
        job = self.job[b, operation]
        start = torch.maximum(self.job_at[b, job], self.machine_at[b, machine])
        finish = start + self.duration[b, operation, machine]
        self.scheduled_machine[b, operation] = machine
        self.start[b, operation], self.finish[b, operation] = start, finish
        self.job_at[b, job], self.machine_at[b, machine] = finish, finish
        self.remaining[b, operation] = False
        self.next[b, job] += 1
        self.remaining_work[b, job] -= self.minimum[b, operation]
        bound = (self.job_at[b] + self.remaining_work[b]).amax(1)
        reward = (self.lower_bound[b] - bound) / self.scale[b]
        self.lower_bound[b] = bound
        self.done = ~self.remaining.any(1)
        return reward, self.done[b]

    def validate(self):
        """Check every duration, compatibility, precedence and machine overlap."""
        for b, inst in enumerate(self.instances):
            machine = self.scheduled_machine[b].cpu().numpy()
            start, finish = self.start[b].cpu().numpy(), self.finish[b].cpu().numpy()
            for o in range(len(inst.duration)):
                m = int(machine[o])
                if m < 0 or inst.duration[o, m] <= 0 or not np.isclose(finish[o] - start[o], inst.duration[o, m]):
                    raise AssertionError('Unscheduled operation, ineligible machine or wrong duration.')
            offset = 0
            for length in inst.lengths:
                if np.any(finish[offset:offset + length - 1] > start[offset + 1:offset + length] + 1e-5):
                    raise AssertionError('Precedence violation.')
                offset += length
            for m in range(inst.duration.shape[1]):
                ops = np.flatnonzero(machine[:len(inst.duration)] == m)
                ops = ops[np.argsort(start[ops])]
                if np.any(finish[ops[:-1]] > start[ops[1:]] + 1e-5):
                    raise AssertionError('Machine overlap.')
        return True
