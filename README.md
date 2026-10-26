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

## Datasets and Environment

Size codes combine the job count with a two-digit machine count: `2010` denotes 20 jobs and 10 machines. Both distributions provide validation sizes `1005`, `2005`, `1510`, and `2010`; test sets additionally include `3010` and `4010`. Training instances are generated online. The environment retains unfinished operations, permits advance machine reservations, and normalizes processing times by each instance's fixed maximum. Assignments are appended to machine schedules, with starts determined by job and machine availability.
Synthetic references are loaded from `or_solution/SD1/solution_<jobs>x<machines>.npy` or `or_solution/SD2/solution_<jobs>x<machines>+mix.npy`. Preserve the complete test folders and original filenames for reference alignment; use `--num-ins` to evaluate a subset. Public evaluation uses `data_test/BenchDataSolution.csv`.

## Training

The supplied `config.json` trains an SD2 model with 15 jobs and 10 machines. Select SD1 or another size through the corresponding command-line options:

```bash
python train.py --distribution SD2 --jobs 20 --machines 10 --device cuda:0 --train_seed 300
```

The supplied configuration uses 5,000 updates, 50 parallel instances, three PPO epochs, learning rate 0.0003, discount factor 1.0, and GAE parameter 0.99. `train_paras.paraller=50` resamples instances every 50 updates; each update collects fresh trajectories. Greedy validation evaluates 100 fixed instances every 50 updates, controlled by `save_timestep`, and selects the lowest mean validation makespan.
`train_paras.minibatch_size=128` controls transition chunks for memory-efficient gradient accumulation. Each epoch accumulates gradients over the complete rollout and takes one optimizer step. Advantages are normalized across the rollout; the critic uses detached pooled encoder features and SmoothL1 loss. ASER updates its exploration coefficient once per rollout.
Run outputs are saved under `train_results/`. JSONL records contain update, policy, value, exploration, loss, structural KL, sampled makespan, and elapsed time; validation records additionally contain current and best validation scores. Use `--config`, `--updates`, `--batch-size`, and `--validation-count` to override run settings.

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

Without `--samples`, `test_paras.sample=false` selects greedy decoding; otherwise `num_sample` controls the sampling budget. A budget of one always selects greedy decoding. The supplied test settings independently select an SD1 model trained on 20 jobs and 10 machines and evaluate `data_test/SD1/3010`. To test the supplied training configuration instead, select SD2 with 15 jobs and 10 machines. `--validate` checks schedule feasibility, while `--batch-size` and `--sample-batch-size` control concurrent instances and samples.
Testing prints the dataset, model path, instance progress, mean makespan, mean instance-wise percentage gap, and elapsed time. Public results also include separate summaries for each benchmark family. `makespan.csv` contains `index`, `file_name`, `makespan`, and `gap (%)`. The mean gap averages each instance's relative deviation from its reference; sampling reports the best schedule per instance.

## Configuration and Reproducibility

FRDT concatenates six operation descriptors (minimum processing time, job availability, mean processing time, compatibility ratio, log-transformed suffix work and operation count), five machine descriptors (availability, mean processing time, log-transformed candidate and remaining-operation counts, and future workload), and four pair descriptors (processing time, future workload, own-job delay, and other-job delay) into separate feature vectors. Three independent linear layers map these vectors to 128-dimensional embeddings, giving a 384-dimensional actor input. Supplementary descriptors are always included; train new checkpoints for this architecture.
The full method enables `model_paras.use_future_demand` and `use_delay_propagation`, with `train_paras.entropy_mode="structural_kl"`. Setting `use_delay_propagation=false` zeros the delay inputs; `use_future_demand=false` zeros future workload and delay inputs while retaining the supplementary state descriptors. Setting `entropy_mode="shannon"` uses fixed-coefficient entropy regularization. Train separate models for ablations.
Seeds are set through `train_seed` and `test_seed` in `config.json`, or `--train_seed` and `--test_seed`. Training and testing request deterministic PyTorch operations. Keep software, hardware, and execution settings consistent for reproducibility. Model selection uses validation data; test instances and reference solutions are reserved for evaluation.

## References

This repository builds on the following open-source implementations:

- https://github.com/XiangjieXiao/ReSched
- https://github.com/wrqccc/FJSP-DRL
- https://github.com/songwenas12/fjsp-drl
- https://github.com/zcaicaros/L2D
- https://github.com/google/or-tools
