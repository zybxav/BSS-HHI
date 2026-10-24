# Running ARBSF

This repository provides a configurable MATLAB entry point at `run_example.m`. The input datasets and output files are not included in the source archive.

## Requirements

- Windows, PowerShell, and MATLAB with support for `-batch` and `maxk`.
- Keep `run_example.m`, `prepare_experiment_state.m`, the other MATLAB source files, and `experiment_runner/` in their original relative locations.

## Prepare input

Create a `datasets/` directory beside `run_example.m`, or use absolute paths to files stored elsewhere. Each nonempty line of a dataset represents one user and contains positive integer item IDs separated by whitespace. A dataset must have at least as many user rows as `config.user_count`.

Edit the configuration section at the top of `run_example.m`. For example:

```matlab
dataset_dir = fullfile(work_dir, 'datasets');
config.dataset_files = {fullfile(dataset_dir, 'my_data.txt')};
config.epsilons = [1, 2];
config.topks = [16, 32];
config.user_count = 100000;
config.repetitions = 5;
```

Add or remove dataset paths as needed. Each dataset file must have a distinct base name, because its output CSV uses that name. The runner evaluates every combination of the supplied privacy budgets and Top-K values for each dataset, repeating each combination `config.repetitions` times. It starts one MATLAB process per dataset. The underlying runner uses `k0 = 10 * topk` and `delay = 5` by default.

## Start the computation

1. Open MATLAB and set its current folder to the directory containing `run_example.m`.
2. Enter `run_example` in the MATLAB Command Window, or run the file from the MATLAB editor.
3. Wait for the launcher to report completion. For a failed process, inspect its log files.

Raw records are appended to `experiment_runner/results/<dataset_base_name>.csv`, one row per repetition. Standard output and errors are written to `experiment_runner/logs/dataset_<index>.stdout.log` and `.stderr.log`, where the index follows the order of `config.dataset_files`. A new run appends to an existing result CSV, so move or remove previous output before starting a fresh run.

