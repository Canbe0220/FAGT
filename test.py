"""Test FAGT models from train_results on SD1, SD2 or public benchmarks."""
import argparse
import copy
import csv
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from FAGT import FAGT
from env.fjsp_env import load_instances
from train import ROOT, model_config, select_device, setup_seed
from validate import distribution_of, problem_code, evaluate

TEST_RESULTS = ROOT / 'test_results'


def resolve_test_path(data_path, distribution):
    """Resolve test data independently of the training-model size."""
    if data_path is None or not str(data_path).strip():
        raise ValueError('Set test_paras.data_path or --data-path, e.g. "3010" or "data_test/SD1/4010".')
    path = Path(str(data_path)).expanduser()
    if not path.is_absolute():
        if len(path.parts) == 1 and path.name.isdigit():
            path = ROOT / 'data_test' / distribution / path
        elif path.parts[0] in ('SD1', 'SD2', 'Public'):
            path = ROOT / 'data_test' / path
        else:
            path = ROOT / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f'Test dataset not found: {path}')
    return path


def load_reference_makespans(instances, env, public=False):
    """Align references to the complete test set before applying --num-ins.

    SD1 rows follow sorted instance filenames; SD2 rows use the explicit
    one-based filename suffix. Sizes come from the data, not the training model.
    Public instances use benchmark/name keys.
    """
    if public:
        source = ROOT / 'data_test' / 'BenchDataSolution.csv'
        if not source.is_file():
            raise FileNotFoundError(f'Reference file not found: {source.relative_to(ROOT)}')
        prefixes = {'brandimarte': 'mk', 'hurink_edata': 'e_la',
                    'hurink_rdata': 'r_la', 'hurink_vdata': 'v_la'}
        reference = {}
        with source.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                benchmark = row['benchname'].strip().lower()
                name = row['dataname'].strip().lower()
                prefix = 'mk' if benchmark == 'brandimarte' else 'la'
                if benchmark not in prefixes or not name.startswith(prefix) or not name[2:].isdigit():
                    raise ValueError('Invalid benchmark or instance name in BenchDataSolution.csv.')
                key = f'{prefixes[benchmark]}{int(name[2:]):02d}'
                if key in reference:
                    raise ValueError(f'Duplicate public reference: {key}')
                reference[key] = float(row['ub'])
        names = [Path(instance.name).stem.lower() for instance in instances]
        missing = [name for name in names if name not in reference]
        if missing:
            raise ValueError(f'No public reference for {missing[0]} in BenchDataSolution.csv.')
        values = np.asarray([reference[name] for name in names], dtype=np.float64)
    else:
        distribution = distribution_of(env)
        sizes = {(len(instance.lengths), instance.duration.shape[1]) for instance in instances}
        if len(sizes) != 1:
            raise ValueError('Synthetic test instances must have one common job/machine size for reference matching.')
        jobs, machines = sizes.pop()
        setting = f'{jobs}x{machines}' + ('+mix' if distribution == 'SD2' else '')
        source = ROOT / 'or_solution' / distribution / f'solution_{setting}.npy'
        if not source.is_file():
            raise FileNotFoundError(f'Reference file not found: {source.relative_to(ROOT)}')
        table = np.load(source, allow_pickle=False)
        if table.ndim != 2 or table.shape != (len(instances), 2):
            raise ValueError(f'{source.relative_to(ROOT)} must contain one [makespan, time] row '
                             f'for each of the {len(instances)} test instances.')
        names = [instance.name for instance in instances]
        if len(set(names)) != len(names):
            raise ValueError('Test instance names must be unique for reference matching.')
        if distribution == 'SD1':
            order = {name: index for index, name in enumerate(sorted(names))}
            indices = [order[name] for name in names]
        else:
            indices = []
            for name in names:
                prefix, _, suffix = Path(name).stem.rpartition('_')
                if prefix != setting or not suffix.isdigit():
                    raise ValueError(f'Cannot match SD2 instance {name} to its reference row.')
                indices.append(int(suffix) - 1)
            if sorted(indices) != list(range(len(instances))):
                raise ValueError('SD2 test instance suffixes must cover all reference rows exactly once.')
        values = np.asarray(table[indices, 0], dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f'Reference makespans in {source.relative_to(ROOT)} must be positive and finite.')
    return values


def select_checkpoint(root, distribution, jobs, machines):
    candidates = (Path(root) / 'train_results').glob('*/best_model.pt')
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime_ns, reverse=True):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        env = checkpoint['config']['env_paras']
        if distribution_of(env) == distribution and env['num_jobs'] == jobs and env['num_mas'] == machines:
            return path
    raise FileNotFoundError(f'No model trained on {distribution} {jobs}x{machines} in '
                            'train_results. Train this setting or supply --model.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config.json'))
    parser.add_argument('--distribution', choices=('SD1', 'SD2'))
    parser.add_argument('--jobs', type=int, help='Job count of the training model to select')
    parser.add_argument('--machines', type=int, help='Machine count of the training model to select')
    parser.add_argument('--data-path', help='Test directory, e.g. 3010, SD1/4010, or data_test/Public')
    parser.add_argument('--model', help='Model file; default selects the latest matching training run')
    parser.add_argument('--public', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--samples', type=int, help='1=greedy; N>1=best of N independent samples')
    parser.add_argument('--sample-batch-size', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--num-ins', type=int)
    parser.add_argument('--test_seed', type=int)
    parser.add_argument('--validate', action='store_true', help='Check every generated schedule for feasibility')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.threads, args.batch_size, args.sample_batch_size) < 1:
        parser.error('Batch sizes and threads must be positive.')
    torch.set_num_threads(args.threads)
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    seed = config.get('test_seed', 100) if args.test_seed is None else args.test_seed
    if seed < 0:
        raise ValueError('seed must be nonnegative.')
    setup_seed(seed)
    test = config['test_paras']
    model_env = copy.deepcopy(config['env_paras'])
    for key, value in (('distribution', args.distribution or test.get('distribution')),
                       ('num_jobs', args.jobs if args.jobs is not None else test.get('num_jobs')),
                       ('num_mas', args.machines if args.machines is not None else test.get('num_mas'))):
        if value is not None:
            model_env[key] = value
    distribution = distribution_of(model_env)
    problem_code(model_env)  # Validate model-selection dimensions only.
    data_path = args.data_path if args.data_path is not None else test.get('data_path')
    # Keep --public as a shortcut. A configured data_path otherwise takes priority.
    if args.public is True or (data_path is None and args.public is not False and test.get('public_ins', False)):
        data_path = 'Public'
    samples = args.samples if args.samples is not None else (test.get('num_sample', 100) if test.get('sample', False) else 1)
    if samples < 1:
        parser.error('--samples must be positive.')
    path = resolve_test_path(data_path, distribution)
    path_label = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()
    public = path.name.lower() == 'public'
    data_distribution = path.parent.name.upper()
    if data_distribution not in ('SD1', 'SD2'):
        data_distribution = distribution
    instances = load_instances(path)
    count = args.num_ins if args.num_ins is not None else min(test.get('num_ins', 100), len(instances))
    if count < 1 or count > len(instances):
        raise ValueError(f'Requested {count} instances; {len(instances)} available in {path_label}.')
    references = load_reference_makespans(instances, {'distribution': data_distribution}, public=public)[:count]
    instances = instances[:count]
    size = problem_code({'num_jobs': len(instances[0].lengths), 'num_mas': instances[0].duration.shape[1]})
    if args.model:
        model_path = Path(args.model).expanduser()
        if not model_path.is_absolute() and not model_path.exists():
            model_path = ROOT / model_path
    else:
        model_path = select_checkpoint(ROOT, distribution, model_env['num_jobs'], model_env['num_mas'])
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    device = select_device(args.device)
    model = FAGT(model_config(checkpoint['config'])).to(device)
    model.load_state_dict(checkpoint['model'])
    model_parts = model_path.resolve().parts
    model_label = (Path(*model_parts[model_parts.index('train_results'):])
                   if 'train_results' in model_parts else Path(model_path.name))
    print(f'Dataset: {path_label}; model: {model_label.as_posix()}', flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    if public:
        scores = np.empty(len(instances), dtype=np.float64)
        for group_name, prefix in (('Mk', 'mk'), ('edata', 'e_la'),
                                   ('rdata', 'r_la'), ('vdata', 'v_la')):
            indices = [i for i, instance in enumerate(instances)
                       if Path(instance.name).stem.lower().startswith(prefix)]
            if not indices:
                continue
            group_started = time.perf_counter()
            group_scores = evaluate(model, [instances[i] for i in indices],
                                    batch_size=args.batch_size, samples=samples,
                                    sample_batch_size=args.sample_batch_size, check=args.validate,
                                    progress=lambda index, ids=indices:
                                    print(f'Test [{ids[index - 1] + 1}] instance', flush=True))
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            group_elapsed = time.perf_counter() - group_started
            scores[indices] = group_scores
            group_gaps = (group_scores - references[indices]) / references[indices] * 100.0
            print(f'{group_name}: instances={len(indices)}, samples={samples}, '
                  f'mean makespan={group_scores.mean():.4f}, '
                  f'mean gap={group_gaps.mean():.4f}%, time={group_elapsed:.2f}s', flush=True)
    else:
        scores = evaluate(model, instances, batch_size=args.batch_size, samples=samples,
                           sample_batch_size=args.sample_batch_size, check=args.validate,
                           progress=lambda index: print(f'Test [{index}] instance', flush=True))
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    gaps = (scores - references) / references * 100.0
    label = 'Public' if public else f'{data_distribution}_{size}'
    output = TEST_RESULTS / f'{label}_{datetime.now():%Y%m%d_%H%M%S_%f}' / 'makespan.csv'
    output.parent.mkdir(parents=True, exist_ok=False)
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['index', 'file_name', 'makespan', 'gap (%)'])
        for index, (instance, score, gap) in enumerate(zip(instances, scores, gaps)):
            writer.writerow([index, instance.name, score, gap])
    print(f'{label}: instances={len(scores)}, samples={samples}, mean makespan={scores.mean():.4f}, '
          f'mean gap={gaps.mean():.4f}%, time={elapsed:.2f}s', flush=True)
    print(f'Saved {output.relative_to(ROOT)}', flush=True)


if __name__ == '__main__':
    main()
