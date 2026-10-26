"""Generate .fjs instances from the configured SD1/SD2 process."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validate import distribution_of, make_case_generator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config.json'))
    parser.add_argument('--distribution', choices=('SD1', 'SD2'))
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--output-dir', help='Defaults to data_generated/<distribution>/<size>')
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error('--count must be positive.')
    env = json.loads(Path(args.config).read_text(encoding='utf-8'))['env_paras']
    if args.distribution:
        env['distribution'] = args.distribution
    distribution = distribution_of(env)
    size = f'{env["num_jobs"]}{env["num_mas"]:02d}'
    output = Path(args.output_dir) if args.output_dir else ROOT / 'data_generated' / distribution / size
    output.mkdir(parents=True, exist_ok=True)
    case = make_case_generator(env)
    for i in range(args.count):
        target = output / f'{distribution}_{size}_{i + 1:03d}.fjs'
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite {target}')
        target.write_text(''.join(case.get_case(i)[0]).strip() + '\n', encoding='utf-8')
    print(f'Created {args.count} {distribution} instances in {output}')


if __name__ == '__main__':
    main()
