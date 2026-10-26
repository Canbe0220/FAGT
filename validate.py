"""SD1/SD2 instance generation, dataset selection and policy evaluation."""
from pathlib import Path
import numpy as np
import torch
from env.case_generator import CaseGenerator_SD2, CaseGenerator_SD1
from env.fjsp_env import FJSPEnv, load_instances

ROOT = Path(__file__).resolve().parent


def distribution_of(env_paras):
    distribution = str(env_paras.get('distribution', 'SD2')).upper()
    if distribution not in ('SD1', 'SD2'):
        raise ValueError('distribution must be SD1 or SD2.')
    return distribution


def problem_code(env_paras):
    jobs, machines = env_paras['num_jobs'], env_paras['num_mas']
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (jobs, machines)):
        raise ValueError('num_jobs and num_mas must be positive integers.')
    return f'{jobs}{machines:02d}'


def dataset_path(split, env_paras, public=False):
    if split not in ('validation', 'test'):
        raise ValueError('split must be validation or test.')
    if public and split != 'test':
        raise ValueError('Public benchmarks are only used for testing.')
    base = ROOT / ('data_dev' if split == 'validation' else 'data_test')
    path = base / 'Public' if public else base / distribution_of(env_paras) / problem_code(env_paras)
    if not path.is_dir():
        raise FileNotFoundError(f'{split.capitalize()} dataset not found: {path.relative_to(ROOT)}')
    return path


def check_instance_size(instances, env_paras, source):
    jobs, machines = env_paras['num_jobs'], env_paras['num_mas']
    if any(len(x.lengths) != jobs or x.duration.shape[1] != machines for x in instances):
        raise ValueError(f'Dataset contains instances outside configured {jobs}x{machines}.')


def make_case_generator(env_paras):
    problem_code(env_paras)
    jobs, machines = env_paras['num_jobs'], env_paras['num_mas']
    if distribution_of(env_paras) == 'SD1':
        return CaseGenerator_SD1(jobs, machines, max(1, int(.8 * machines)),
                                 max(1, int(1.2 * machines)), flag_same_opes=False)
    # CaseGenerator_SD2 uses mixed machine eligibility by default.
    return CaseGenerator_SD2(jobs, machines, machines, machines)


def get_validate_env(env_paras):
    size = env_paras.get('valid_batch_size', 100)
    if size < 1:
        raise ValueError('Validation count must be positive.')
    path = dataset_path('validation', env_paras)
    instances = load_instances(path)
    if len(instances) < size:
        raise ValueError(f'Only {len(instances)} validation instances in {path.relative_to(ROOT)}; requested {size}.')
    instances = instances[:size]
    check_instance_size(instances, env_paras, path)
    print(f'Validation: {path.relative_to(ROOT)} ({len(instances)} instances).', flush=True)
    return instances


@torch.no_grad()
def evaluate(model_policy, instances, batch_size=16, samples=1,
             sample_batch_size=10, check=False, progress=None):
    """Decode schedules; optionally report each completed instance (one-based)."""
    if not instances or min(batch_size, samples, sample_batch_size) < 1:
        raise ValueError('Instances and evaluation batch sizes must be nonempty.')
    previous_mode = model_policy.training
    model_policy.eval()
    device = next(model_policy.parameters()).device
    scores = []
    try:
        for start in range(0, len(instances), batch_size):
            group = instances[start:start + batch_size]
            best = np.full(len(group), np.inf)
            for sample_start in range(0, samples, sample_batch_size):
                copies = min(sample_batch_size, samples - sample_start)
                env = FJSPEnv([x for x in group for _ in range(copies)], device)
                while not env.done.all():
                    obs = env.observe()
                    action, _, _ = model_policy.act(obs, greedy=samples == 1)
                    env.step(action, obs)
                if check:
                    env.validate()
                score = env.makespan.cpu().numpy().reshape(len(group), copies).min(1)
                best = np.minimum(best, score)
            scores.extend(best.tolist())
            if progress is not None:
                for index in range(start + 1, start + len(group) + 1):
                    progress(index)
    finally:
        model_policy.train(previous_mode)
    return np.asarray(scores, dtype=np.float64)
