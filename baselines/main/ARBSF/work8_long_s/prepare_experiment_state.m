function state = prepare_experiment_state(filename, k0, k2, epsilon, user_count)
% Build the two-stage experiment state without constructing a padded full matrix.

file_id = fopen(filename, 'r');
if file_id == -1
    error('Cannot open input dataset: %s', filename);
end
cleanup = onCleanup(@() fclose(file_id));
lines = cell(0, 1);
max_value = 0;
while ~feof(file_id)
    line = fgetl(file_id);
    if ischar(line)
        values = sscanf(line, '%d').';
        if ~isempty(values)
            lines{end + 1, 1} = values; %#ok<AGROW>
            max_value = max(max_value, max(values));
        end
    end
end
clear cleanup

available_rows = numel(lines);

if user_count > available_rows
    error('请求用户数 %d 超过数据集总行数 %d。', ...
        user_count, available_rows);
end

% 从 600k 数据中随机抽取本轮实验所需用户数
lines = lines(randperm(available_rows, user_count));
num_rows = user_count;

if num_rows < 2 || max_value < k0
    error('Dataset %s does not contain enough用户或不同的 item ID。', filename);
end

shuffled_lines = lines(randperm(num_rows));
first_count = floor(num_rows / 2);
selected = zeros(first_count, 1);
for row_index = 1:first_count
    values = shuffled_lines{row_index};
    selected(row_index) = values(randi(numel(values)));
end

selected_counts = accumarray(selected, 1, [max_value, 1]);
signal_items = selected(rand(first_count, 1) < 0.5);
signal_counts = accumarray(signal_items, 1, [max_value, 1]);
q = 1 / (exp(epsilon) + 1);
noise_mean = (first_count - selected_counts) * q;
noise_std = sqrt((first_count - selected_counts) * q * (1 - q));
noise_counts = round(noise_mean + noise_std .* randn(max_value, 1));
noise_counts = min(max(noise_counts, 0), first_count - selected_counts);
phase1_counts = signal_counts + noise_counts;
[~, indices] = maxk(phase1_counts, k0);

frequency = zeros(max_value, 1, 'uint32');
for row_index = 1:num_rows
    values = shuffled_lines{row_index};
    [items, ~, item_index] = unique(values);
    item_counts = accumarray(item_index(:), 1);
    frequency(items) = frequency(items) + uint32(item_counts);
end
[ground_counts, ground_items] = maxk(double(frequency), k2);
Ground_truth = [ground_items, ground_counts];

second_start = floor(num_rows / 2);
second_count = num_rows - second_start + 1;
present = false(max_value, 1);
for row_index = 1:second_count
    values = shuffled_lines{second_start + row_index - 1};
    kept = values(ismember(values, indices));
    present(kept) = true;
end
unique_values = find(present);
d2 = numel(unique_values);
if d2 < k2
    error('Only %d Phase-1 candidates occur in Phase 2; need k=%d.', d2, k2);
end

item = zeros(second_count, d2, 'uint8');
for row_index = 1:second_count
    values = shuffled_lines{second_start + row_index - 1};
    [is_present, locations] = ismember(values, unique_values);
    item(row_index, locations(is_present)) = 1;
end

phase1_hits = numel(intersect(Ground_truth(:, 1), indices));
fprintf('Phase1 top-k accuracy = %.4f  (hits = %d / %d)\n', phase1_hits / k2, phase1_hits, k2);
state = struct('num_rows', num_rows, 'd2', d2, 'item', item, ...
    'Ground_truth', Ground_truth, 'unique_values', unique_values);
end
