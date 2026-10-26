"""Train FAGT with FRDT and ASER on SD1 or SD2."""
import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from FAGT import FAGT, PPOTrainer
from env.fjsp_env import generate_case_instances
from validate import distribution_of, evaluate, get_validate_env, make_case_generator, problem_code

ROOT = Path(__file__).resolve().parent
TRAIN_RESULTS = ROOT / 'train_results'


def setup_seed(seed):
    """Seed all RNGs and require deterministic operations on the same setup."""
    # Configure cuBLAS before initializing CUDA or constructing the model.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def select_device(label):
    if label == 'auto':
        # Prefer the third GPU when available; otherwise use the first GPU.
        label = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    device = torch.device(label)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available in PyTorch.')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f'CUDA index {device.index} is outside the visible device range.')
        torch.cuda.set_device(device)
    return device


def model_config(config):
    p = config['model_paras']
    if p.get('dropout', 0) != 0:
        raise ValueError('Dropout must be zero for consistent PPO log probabilities.')
    keys = ('embedding_dim', 'head_num', 'block_num', 'ff_hidden_dim', 'use_future_demand',
            'use_delay_propagation',
            'actor_hidden_dim', 'critic_hidden_dim', 'actor_layers', 'critic_layers')
    return {key: p[key] for key in keys if key in p}


def ppo_config(config, batch_size):
    p = config['train_paras']
    keys = ('lr', 'betas', 'gamma', 'gae_lambda', 'K_epochs', 'entropy_coeff',
            'entropy_mode', 'kl_target', 'micro_batch_size', 'max_grad_norm')
    result = {key: p[key] for key in keys if key in p}
    result.update(clip_range=p.get('eps_clip', 0.2), vf_coef=p.get('vf_coeff', 0.5),
                  policy_coef=p.get('A_coeff', 1.0), minibatch_size=p.get('minibatch_size', 64))
    return result


def save_model(model, config, path):
    """Save FP32 model weights and configuration for evaluation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    temporary = path.with_suffix('.tmp')
    torch.save({'model': weights, 'config': config}, temporary)
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config.json'))
    parser.add_argument('--distribution', choices=('SD1', 'SD2'))
    parser.add_argument('--jobs', type=int)
    parser.add_argument('--machines', type=int)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--updates', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--validation-count', type=int)
    parser.add_argument('--train_seed', type=int)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1 or (args.train_seed is not None and args.train_seed < 0):
        parser.error('--threads must be positive and --train_seed nonnegative.')
    torch.set_num_threads(args.threads)
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    env_config, training = config['env_paras'], config['train_paras']
    for key, value in (('distribution', args.distribution), ('num_jobs', args.jobs),
                       ('num_mas', args.machines), ('batch_size', args.batch_size),
                       ('valid_batch_size', args.validation_count)):
        if value is not None:
            env_config[key] = value
    if args.updates is not None:
        training['max_iterations'] = args.updates
    distribution = distribution_of(env_config)
    env_config['distribution'] = distribution
    size = problem_code(env_config)
    batch_size = env_config['batch_size']
    updates, interval = training['max_iterations'], training['save_timestep']
    resample_interval = training.setdefault('paraller', 50)
    if isinstance(resample_interval, bool) or not isinstance(resample_interval, int) or resample_interval < 1:
        raise ValueError('train_paras.paraller must be a positive integer.')
    if min(batch_size, env_config['valid_batch_size'], updates, interval) < 1:
        raise ValueError('Batch sizes, max_iterations and save_timestep must be positive.')
    if updates < interval:
        raise ValueError('max_iterations must include at least one validation interval (save_timestep).')
    seed = config.get('train_seed', 300) if args.train_seed is None else args.train_seed
    if seed < 0:
        raise ValueError('train_seed must be nonnegative.')
    config['train_seed'] = seed
    setup_seed(seed)
    device = select_device(args.device)
    model = FAGT(model_config(config)).to(device)
    trainer = PPOTrainer(model, ppo_config(config, batch_size))
    validation = get_validate_env(env_config)
    case = make_case_generator(env_config)
    output = TRAIN_RESULTS / f'{distribution}_{size}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'FAGT / FRDT / ASER; {distribution} {env_config["num_jobs"]}x{env_config["num_mas"]}; '
          f'device={device}; updates={updates}; batch={batch_size}; validation every {interval} updates.', flush=True)
    print(f'Training results: {output.relative_to(ROOT)}', flush=True)
    best = float('inf')
    training_instances = None
    for step in range(1, updates + 1):
        started = time.perf_counter()
        if (step - 1) % resample_interval == 0:
            training_instances = generate_case_instances(case, batch_size)
        # Reuse instances, but collect a fresh policy rollout at every update.
        metrics = trainer.train_batch(training_instances)
        score = None
        if step % interval == 0:
            score = float(evaluate(model, validation).mean())
            if not np.isfinite(score):
                raise FloatingPointError('Validation produced a nonfinite makespan.')
            if score < best:
                best = score
                save_model(model, config, output / 'best_model.pt')
        record = {'update': step, **metrics, 'times': time.perf_counter() - started}
        if score is not None:
            record.update(validation=score, best_validation=best)
        with (output / 'training.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(record, allow_nan=False) + '\n')
        print(json.dumps({key: value for key, value in record.items()
                          if key not in ('validation', 'best_validation')}, allow_nan=False), flush=True)
        if score is not None:
            print(json.dumps({'validation': score, 'best_validation': best}, allow_nan=False), flush=True)
    print(f'Saved {output.relative_to(ROOT) / "best_model.pt"}', flush=True)


if __name__ == '__main__':
    main()
