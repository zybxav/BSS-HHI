dbstop if error
format short
opts = optimoptions('fsolve', 'Display', 'off');
filename1 = 'guassian_input.txt';
% k2=13; % 举例，找出频率最高的5个元素
%ground_truth_sta
% data_dispove;

% 设置 'data1' 文件夹的路径
data_folder = 'data\guassian';

% 获取 'data1' 文件夹中的所有文件
file_info = dir(fullfile(data_folder, '*.txt'));

% 遍历每个文件
for i = 1:length(file_info)
    % 构建完整的文件名
    filename = fullfile(data_folder, file_info(i).name);
    
    % 在这里运行 'hello.m' 代码
    disp(['Processing file: ', filename]);
    run('hello_beta25.m');
end

hello_beta25;
musicc