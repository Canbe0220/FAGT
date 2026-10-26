# FAGT

This is the anonymous implementation of **"Future-Aware Graph Transformers with Adaptive Exploration for Flexible Job-Shop Scheduling"**.

FAGT learns joint operation selection and machine assignment to minimize makespan in the flexible job-shop scheduling problem (FJSP). It represents unfinished operations and machines as a graph and integrates two complementary modules into proximal policy optimization (PPO):

- **Future-Aware Resource Dependency Transformer (FRDT):** combines directed successor attention, future workload aggregation, and relaxed delay propagation to encode the downstream consequences of candidate assignments within and across jobs.
- **Adaptive Structured Entropy Regularization (ASER):** uses normalized KL divergence from the policy to a job-balanced reference distribution and adapts its coefficient to regulate exploration under unequal machine eligibility.

## Requirements

Python 3.10 or later, PyTorch, and NumPy are required. Run the following command in your Python environment:

```bash
python -m pip install torch numpy
```

For GPU execution, install a PyTorch build compatible with your CUDA environment. All commands below explicitly select `cuda:0`; replace it with `cpu` or another visible CUDA device as needed. Run commands from the repository root.

## Repository Structure

| Path | Description |
| --- | --- |
| `FAGT.py` | Actor and critic, rollout collection, generalized advantage estimation (GAE), and PPO optimization. |
| `FRDT.py` | Graph Transformer encoder, successor attention, future resource demand, and relaxed delay propagation. |
| `ASER.py` | Job-balanced reference probabilities, normalized structural KL regularization, and coefficient adaptation. |
| `train.py` | Online training, periodic validation, logging, and checkpoint selection. |
| `validate.py` | Validation dataset loading and shared greedy or sampling evaluation. |
| `test.py` | Checkpoint selection, synthetic/public benchmark testing, reference matching, and CSV export. |
| `config.json` | Dataset, network, training, testing, and random-seed settings. |
| `config_annotation.json` | Configuration-field descriptions; runtime settings are defined by `config.json` and the Python modules. |
| `mlp.py` | Supporting multilayer perceptron definitions. |
| `env/` | Instance generation, parsing, and scheduling environment package. |
| `env/fjsp_env.py` | Instance representations, data loading, state transitions, normalization, rewards, and feasibility checks. |
| `env/case_generator.py` | Online SD1 and SD2 instance generators used during training. |
| `env/load_data.py` | Supporting FJS parsing and operation-matrix construction helpers. |
| `env/__init__.py` | Environment package initialization. |
| `utils/` | Auxiliary dataset-generation, memory-inspection, and JSON utilities. |
| `utils/create_ins.py` | Generates standalone FJS datasets from the configured SD1 or SD2 distribution. |
| `utils/gpu_mem_track.py` | Inspects GPU tensor allocations and memory usage. |
| `utils/modelsize_estimate.py` | Estimates model parameter and intermediate-tensor memory requirements. |
| `utils/my_utils.py` | JSON reading and writing helpers. |
| `utils/color_config.json` | Supporting color definitions. |
| `data_dev/` | Fixed validation instances in `SD1/<size>/` and `SD2/<size>/` subdirectories. |
| `data_test/` | Synthetic test instances in `SD1/<size>/` and `SD2/<size>/`, plus public benchmarks. |
| `data_test/Public/` | Brandimarte and Hurink edata, rdata, and vdata instances. |
| `data_test/BenchDataSolution.csv` | Public benchmark references; testing uses the `ub` column. |
| `or_solution/` | Synthetic reference arrays in `SD1/` and `SD2/`, with additional per-instance files in size subdirectories. |
| `or_solution/BenchData/` | Archived solver results grouped into `Brandimarte/`, `Hurink_edata/`, `Hurink_rdata/`, and `Hurink_vdata/`. |
| `train_results/<run>/` | Generated training outputs: `config.json` stores run settings, `training.jsonl` stores update metrics, and `best_model.pt` stores selected weights and configuration. |
| `test_results/<dataset>_<timestamp>/` | Generated evaluation outputs; `makespan.csv` records each instance's makespan and percentage gap. |
| `data_generated/` | Default output directory created by `utils/create_ins.py`, organized by distribution and size. |
| `README.md` | Project overview, repository layout, and execution instructions. |
| `LICENSE` | Repository license terms. |

## Training

The supplied `config.json` trains an SD2 model with 15 jobs and 10 machines. Select SD1 or another size through the corresponding command-line options:

```bash
python train.py --distribution SD2 --jobs 20 --machines 10 --device cuda:0 --train_seed 300
```

## Testing

`--distribution`, `--jobs`, and `--machines` select the latest matching checkpoint under `train_results/`; `--model` selects an explicit checkpoint. `--data-path` independently selects the test dataset. Instance dimensions are read from its files, enabling evaluation on larger sizes without changing the model-selection dimensions.

```bash
# Greedy decoding with an SD1 20-by-10 checkpoint
python test.py --distribution SD1 --jobs 20 --machines 10 --data-path 2010 --samples 1 --device cuda:0 --validate
# Best-of-100 sampling
python test.py --distribution SD1 --jobs 20 --machines 10 --data-path 2010 --samples 100 --device cuda:0 --validate
# Cross-size evaluation using the same 20-by-10 model
python test.py --distribution SD1 --jobs 20 --machines 10 --data-path 3010 --samples 1 --device cuda:0 --validate
python test.py --distribution SD1 --jobs 20 --machines 10 --data-path 4010 --samples 1 --device cuda:0 --validate
# All public benchmarks: 10 Brandimarte and 40 instances per Hurink family
python test.py --distribution SD1 --jobs 20 --machines 10 --data-path Public --samples 1 --num-ins 130 --device cuda:0 --validate
```


## References

This repository builds on the following open-source implementations:

- https://github.com/XiangjieXiao/ReSched
- https://github.com/wrqccc/FJSP-DRL
- https://github.com/songwenas12/fjsp-drl
- https://github.com/zcaicaros/L2D
- https://github.com/google/or-tools
