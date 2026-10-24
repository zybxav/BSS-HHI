% Configure datasets and parameters here, then run this script in MATLAB.
work_dir = fileparts(mfilename('fullpath'));
dataset_dir = fullfile(work_dir, 'datasets');
config.dataset_files = { ...
    fullfile(dataset_dir, 'your_dataset_input.txt')};

config.epsilons = [1, 2];
config.topks = [16, 32];
config.user_count = 100000;
config.repetitions = 5;
% run_experiment defaults to k0 = 10 * topk and delay = 5.

if ~iscell(config.dataset_files) || isempty(config.dataset_files)
    error('run_example:Datasets', 'Provide at least one dataset path.');
end
dataset_names = cell(1, numel(config.dataset_files));
for dataset_index = 1:numel(config.dataset_files)
    if ~isfile(config.dataset_files{dataset_index})
        error('run_example:MissingDataset', 'Dataset not found: %s', config.dataset_files{dataset_index});
    end
    [~, dataset_names{dataset_index}] = fileparts(config.dataset_files{dataset_index});
end
if numel(unique(dataset_names)) ~= numel(dataset_names)
    error('run_example:DatasetNames', 'Dataset basenames must be unique.');
end
validateattributes(config.epsilons, {'numeric'}, {'vector','nonempty','real','finite','positive'});
validateattributes(config.topks, {'numeric'}, {'vector','nonempty','real','finite','integer','positive'});
validateattributes(config.user_count, {'numeric'}, {'scalar','integer','>=',2});
validateattributes(config.repetitions, {'numeric'}, {'scalar','integer','positive'});

runner_dir = fullfile(work_dir, 'experiment_runner');
launcher = fullfile(runner_dir, 'launch_four_datasets.ps1');
if ~isfile(launcher) || ~isfile(fullfile(runner_dir, 'run_dataset_experiments.m'))
    error('run_example:MissingRunner', 'Runner files not found: %s', runner_dir);
end

config_file = [tempname, '.mat'];
save(config_file, 'config');
config_cleanup = onCleanup(@() delete(config_file)); %#ok<NASGU>
matlab_exe = fullfile(matlabroot, 'bin', 'matlab.exe');
command = sprintf(['powershell.exe -NoProfile -ExecutionPolicy Bypass ' ...
    '-File "%s" -MatlabExe "%s" -RunnerDir "%s" -ConfigFile "%s" ' ...
    '-DatasetCount %d'], ...
    launcher, matlab_exe, runner_dir, config_file, numel(config.dataset_files));
fprintf('Starting %d dataset process(es). Logs: %s\n', ...
    numel(config.dataset_files), fullfile(runner_dir, 'logs'));
status = system(command);
clear config_cleanup
if status ~= 0
    error('run_example:ParallelFailed', 'A process failed. Inspect experiment_runner/logs.');
end
fprintf('All datasets finished. Results: %s\n', fullfile(runner_dir, 'results'));
