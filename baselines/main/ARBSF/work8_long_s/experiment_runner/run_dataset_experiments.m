function result_file = run_dataset_experiments(dataset_index, config_file, dry_run)
%RUN_DATASET_EXPERIMENTS Run one configurable sweep for one dataset.
if nargin < 3
    dry_run = false;
end
loaded = load(config_file, 'config');
config = loaded.config;
validateattributes(dataset_index, {'numeric'}, ...
    {'scalar', 'integer', '>=', 1, '<=', numel(config.dataset_files)});

runner_dir = fileparts(mfilename('fullpath'));
dataset_file = config.dataset_files{dataset_index};
if ~isfile(dataset_file)
    error('run_dataset_experiments:MissingDataset', 'Dataset not found: %s', dataset_file);
end
addpath(runner_dir);
[~, dataset_name] = fileparts(dataset_file);
result_file = fullfile(runner_dir, 'results', [dataset_name, '.csv']);
if dry_run
    fprintf(['READY dataset=%s result=%s users=%d repetitions=%d ' ...
        'eps=%s k=%s\n'], ...
        dataset_file, result_file, config.user_count, config.repetitions, ...
        mat2str(config.epsilons), mat2str(config.topks));
    return;
end

for epsilon = config.epsilons
    for topk = config.topks
        run_config = struct('filename', dataset_file, 'epsilon', epsilon, ...
            'k2', topk, 'user_counts', config.user_count, ...
            'repetitions', config.repetitions, 'experiment', 'sweep', ...
            'result_file', result_file);
        run_experiment(run_config);
    end
end
fprintf('Dataset finished: %s\n', result_file);
end
