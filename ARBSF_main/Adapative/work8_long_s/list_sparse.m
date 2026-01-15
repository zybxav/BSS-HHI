clear all;
format short
opts = optimoptions('fsolve', 'Display', 'off');

% 参数设置
delay = 5;
epsilon = 2;
k0 = 300;
k2 = 30;

% 文件名列表
filenames = {
    'datasparse/gamma/gamma_s2.txt',
    'datasparse/gamma/gamma_s4.txt',
    'datasparse/gamma/gamma_s5.txt',
    'datasparse/gamma/gamma_s7.txt',
    'datasparse/gamma/gamma_s9.txt',
    'datasparse/laplace/laplace_s2.txt',
    'datasparse/laplace/laplace_s4.txt',
    'datasparse/laplace/laplace_s5.txt',
    'datasparse/laplace/laplace_s7.txt',
    'datasparse/laplace/laplace_s9.txt'
};

% 启动并行池（如果尚未启动）
if isempty(gcp('nocreate'))
    parpool;
end

% 预加载所有文件数据以避免透明性问题
data = cell(size(filenames));
for i = 1:length(filenames)
    data{i} = load(filenames{i}); % 假设文件内容可以通过 load 函数加载
end

% 并行处理文件
parfor i = 1:length(filenames)
    filename = filenames{i};
    filedata = data{i}; % 获取预加载的数据

    % heihei 的内容
    delay = 5;
    epsilon = 2;
    k0 = 300;
    k2 = 30;

    % 读取 ground_truth_sta 的内容
    % 假设 ground_truth_sta 是一个脚本，我们把内容直接放到这里
    % ground_truth_sta 内容开始
    % 例如：
    % num_rows = ...; % 从 filedata 计算或提取
    % d2 = ...; % 从 filedata 计算或提取
    % item = ...; % 从 filedata 计算或提取
    % Ground_truth = ...; % 从 filedata 计算或提取
    % unique_values = ...; % 从 filedata 计算或提取
    % ground_truth_sta 内容结束

    % 计算 N
    N = num_rows - floor(num_rows / 2) + 1;

    % 调用 phase2_pre
    phase2_pre(d2, item, N, k2, delay, Ground_truth, unique_values, filename, epsilon);
end

musicc();