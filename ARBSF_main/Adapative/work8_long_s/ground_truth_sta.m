
% 读取文件内容，假设它全部由整数构成
% 读取文件内容，假设它全部由整数构成
%改：修改读取逻辑，应该将所有用户的矩阵读入，但是读入出问题了，改为最大size，然后没数值填0

% 逐行读取
fid = fopen(filename, 'r');
lines = {};
while ~feof(fid)
    line = fgetl(fid);
    if ischar(line)
        % 按空格切分并转换为数字
        nums = str2double(strsplit(strtrim(line)));
        lines{end+1} = nums;
    end
end
fclose(fid);

% 找到最大列数
max_len = max(cellfun(@length, lines));

% 构造补零后的矩阵
data = zeros(length(lines), max_len, 'int32');

for i = 1:length(lines)
    row = lines{i};
    data(i, 1:length(row)) = row;
end






data_matrix=data;
num_rows = size(data_matrix, 1);
rand_indices = randperm(num_rows);
    
% 使用随机索引重新排序矩阵的行
shuffled_matrix = data_matrix(rand_indices, :);

shuffled_half1=shuffled_matrix(1:floor(num_rows/2),:);
shuffled_half2=shuffled_matrix(floor(num_rows/2):num_rows,:);
%pick_nonzero_and_add
selected_elements = pick_nonzero_and_add(shuffled_half1);
max_value = max(max(data_matrix));

%%%%%%%%%%%%%%%phaseI找出频率最高的数据

vector_sum=zeros(1,max_value);
for i = 1:length(selected_elements)
    temp = zeros(1, max_value);

    % 假设 selected_elements{i} 是一个数组（多个位置）
    temp(selected_elements(i)) = 1;

    binary_str = OUE(temp, epsilon);
    vector_sum = vector_sum + binary_str;

    % % ---- 打印多个位置 ----
    % idx = find(temp == 1);
    % 
    % % 打印成 [a b c] 的格式
    % fprintf('第 %d 次循环: temp 中为 1 的位置 = [', i);
    % fprintf('%d ', idx);
    % fprintf(']\n');
end

[~, indices] = maxk(vector_sum, k0);

%%%%将剩余的拥有topk的内容保留下来，
%此处得到的shuffled_half_final是第二组用户在top-k0中的元素的保留，对于不在top-k0中的元素进行了归零，然后删去了多余的0列，最后是50001*136
shuffled_half_final = process_matrix(shuffled_half2, indices);

data_dispove;

%%%%%%%计算groud_truth
% 将数据向量化（如果是多维的话）
data = shuffled_matrix;

% 移除所有的0值
data(data == 0) = [];

% 计算每个唯一元素的频率
[uniqueElements, ~, idx] = unique(data);
frequency = accumarray(idx, 1);

uniqueElements = uniqueElements(:);   % 转为列向量
frequency = frequency(:);             % 确保是列向量



% 创建一个数组来存储元素和其频率
elementFrequency = [uniqueElements, frequency];


% 按频率降序排序
[~, sortIdx] = sort(elementFrequency(:, 2), 'descend');
sortedMatrix = elementFrequency(sortIdx, :);

% 用户定义想要的元素个数k

% 选择频率最高的k个元素(第二阶段的k2个，此处是30)
topKElements = sortedMatrix(1:k2, :);
Ground_truth = topKElements;

test=length(intersect(Ground_truth(:,1),indices));
true_topk = Ground_truth(:,1);   % 真正的 top-k2 items
pred_topk = indices;             % Phase1 估计的 top-k0 items

intersection_items = intersect(true_topk, pred_topk);
hit_count = length(intersection_items);   % 交集大小

k2_size = length(true_topk);             % 分母 = k2

phase1_accuracy = hit_count / k2_size;

fprintf('Phase1 top-k accuracy = %.4f  (hits = %d / %d)\n', ...
        phase1_accuracy, hit_count, k2_size);
