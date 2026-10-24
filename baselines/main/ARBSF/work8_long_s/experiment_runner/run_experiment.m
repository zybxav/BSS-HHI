function output_dirs = run_experiment(config, varargin)
%RUN_EXPERIMENT Parameterized entry for the kos_all.m experiment.
% Use run_experiment(filename, epsilon, topk, user_count).
% A config struct is also supported for advanced settings.

runner_dir = fileparts(mfilename('fullpath'));
work_dir = fileparts(runner_dir);
if ~isstruct(config)
    if numel(varargin) < 3 || numel(varargin) > 4
        error('run_experiment:Arguments', ...
            'Use config or filename, epsilon, k2, user_count, [repetitions].');
    end
    filename = config;
    config = struct('filename', filename, 'epsilon', varargin{1}, ...
        'k2', varargin{2}, 'user_counts', varargin{3});
    if numel(varargin) == 4
        config.repetitions = varargin{4};
    end
end
if ~isscalar(config)
    error('run_experiment:Arguments', 'config must be a scalar struct.');
end

config = default_field(config, 'code_root', work_dir);
config = default_field(config, 'dataset_name', '');
config = default_field(config, 'result_root', fullfile(runner_dir, 'results'));
config = default_field(config, 'repetitions', 5);
config = default_field(config, 'delay', 5);
config = default_field(config, 'result_file', '');
config = default_field(config, 'experiment', '');
if isfield(config, 'k2')
    config = default_field(config, 'k0', 10 * config.k2);
end
required = {'filename', 'epsilon', 'k2', 'k0', 'user_counts'};
for i = 1:numel(required)
    if ~isfield(config, required{i})
        error('run_experiment:MissingParameter', 'Missing config.%s.', required{i});
    end
end
validate_text(config.filename, 'filename');
validate_text(config.code_root, 'code_root');
validate_text(config.result_root, 'result_root');
if ~isempty(config.dataset_name)
    validate_text(config.dataset_name, 'dataset_name');
end
if ~isempty(config.result_file)
    validate_text(config.result_file, 'result_file');
    validate_text(config.experiment, 'experiment');
end
validateattributes(config.epsilon, {'numeric'}, {'scalar', 'real', 'finite', 'positive'});
validateattributes(config.k2, {'numeric'}, {'scalar', 'real', 'finite', 'integer', 'positive'});
validateattributes(config.k0, {'numeric'}, {'scalar', 'real', 'finite', 'integer', 'positive'});
validateattributes(config.delay, {'numeric'}, {'scalar', 'real', 'finite', 'integer', 'positive'});
validateattributes(config.repetitions, {'numeric'}, {'scalar', 'real', 'finite', 'integer', 'positive'});
validateattributes(config.user_counts, {'numeric'}, ...
    {'vector', 'nonempty', 'real', 'finite', 'integer', '>=', 2});
if config.k0 < config.k2
    error('run_experiment:InvalidK', 'k0 must be at least k2.');
end

filename = char(config.filename);
if ~isfile(filename)
    error('run_experiment:MissingDataset', 'Dataset not found: %s', filename);
end
if isempty(fileparts(filename))
    filename = fullfile(pwd, filename);
end
code_root = char(config.code_root);
if ~isfolder(code_root) || ~isfile(fullfile(code_root, 'prepare_experiment_state.m'))
    error('run_experiment:MissingHelper', ...
        'code_root or prepare_experiment_state.m is missing: %s', code_root);
end
old_path = path;
path_cleanup = onCleanup(@() path(old_path)); %#ok<NASGU>
addpath(work_dir, code_root);

dataset_name = char(config.dataset_name);
if isempty(dataset_name)
    [~, dataset_name] = fileparts(filename);
end
dataset_name = regexprep(dataset_name, '[^a-zA-Z0-9_-]', '_');
result_root = char(config.result_root);
global ARBSF_OUTPUT_DIR ARBSF_RUN_ID;
previous_output = ARBSF_OUTPUT_DIR;
previous_run = ARBSF_RUN_ID;
global_cleanup = onCleanup(@() restore_globals(previous_output, previous_run)); %#ok<NASGU>

output_dirs = cell(size(config.user_counts));
for index = 1:numel(config.user_counts)
    user_count = config.user_counts(index);
    output_dir = fullfile(result_root, dataset_name, sprintf('n%d', user_count), ...
        sprintf('eps_%g', config.epsilon), sprintf('k%d_k0%d_delay%d', ...
        config.k2, config.k0, config.delay));
    if ~isempty(config.result_file)
        output_dir = char(config.result_file);
    end
    output_dirs{index} = output_dir;
    for run_id = 1:config.repetitions
        fprintf(['\n[Experiment] dataset=%s users=%d repeat=%d/%d ' ...
            'k2=%d k0=%d epsilon=%g delay=%d\n'], dataset_name, ...
            user_count, run_id, config.repetitions, config.k2, config.k0, ...
            config.epsilon, config.delay);
        state = prepare_experiment_state(filename, config.k0, config.k2, ...
            config.epsilon, user_count);
        N = size(state.item, 1);
        if N > state.num_rows
            error('run_experiment:InvalidState', ...
                'Phase 2 has %d rows, but only %d users were selected.', N, state.num_rows);
        end
        if isempty(config.result_file)
            if ~isfolder(output_dir)
                mkdir(output_dir);
            end
            ARBSF_OUTPUT_DIR = output_dir;
            ARBSF_RUN_ID = run_id;
            phase2_pre(state.d2, state.item, N, config.k2, config.delay, ...
                state.Ground_truth, state.unique_values, filename, config.epsilon);
        else
            part = n0_caculate(N, state.d2, config.epsilon);
            part = min(part, N / state.d2);
            [ncr, mse, acc, f1, mcr, ~, pred, truth] = phase2_process( ...
                state.d2, config.epsilon, state.item, N, config.k2, ...
                state.Ground_truth, part, config.delay, 1, state.unique_values);
            append_csv_result(output_dir, config.experiment, dataset_name, ...
                user_count, N, config.epsilon, config.k2, config.k0, ...
                config.delay, run_id, ncr, mse, acc, f1, mcr, pred, truth);
        end
    end
    fprintf('Results: %s\n', output_dir);
end
if isscalar(output_dirs)
    output_dirs = output_dirs{1};
end
end

function config = default_field(config, name, value)
if ~isfield(config, name)
    config.(name) = value;
end
end

function validate_text(value, name)
if ~((ischar(value) && isrow(value)) || (isstring(value) && isscalar(value))) ...
        || strlength(string(value)) == 0
    error('run_experiment:InvalidText', '%s must be a nonempty text value.', name);
end
end

function restore_globals(output_dir, run_id)
global ARBSF_OUTPUT_DIR ARBSF_RUN_ID;
ARBSF_OUTPUT_DIR = output_dir;
ARBSF_RUN_ID = run_id;
end

function append_csv_result(file, experiment, dataset, users, phase2_users, ...
        epsilon, k, k0, delay, run_id, ncr, mse, acc, f1, mcr, pred, truth)
parent = fileparts(file);
if ~isfolder(parent)
    mkdir(parent);
end
new_file = ~isfile(file) || dir(file).bytes == 0;
fid = fopen(file, 'a');
if fid < 0
    error('run_experiment:Output', '无法写入结果文件：%s', file);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
if new_file
    fprintf(fid, ['experiment,dataset,user_count,phase2_users,epsilon,k,k0,delay,' ...
        'run,ncr,mse,acc,f1,mcr,predicted_topk,ground_truth_topk\n']);
end
fprintf(fid, '%s,%s,%d,%d,%.17g,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%s,%s\n', ...
    experiment, dataset, users, phase2_users, epsilon, k, k0, delay, run_id, ...
    ncr, mse, acc, f1, mcr, format_pairs(pred), format_pairs(truth));
end

function value = format_pairs(pairs)
value = '';
for i = 1:size(pairs, 1)
    if i > 1
        value = [value, ';']; %#ok<AGROW>
    end
    value = [value, sprintf('%d:%.17g', pairs(i, 1), pairs(i, 2))]; %#ok<AGROW>
end
end
