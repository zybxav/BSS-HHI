dbstop if error
%data_generate_design
%d2=d;
% 文件名
filename = 'kosarak_input_ep1.0_k90.txt';

% 读取文件到变量data中
% 假设数据是用空格或逗号分隔的，需要根据实际情况选择适合的函数和参数
data = readmatrix(filename, 'Delimiter', ' ');
data2 = data;
data1 = data(~isnan(data));

% 使用unique函数找出所有不同的数值
unique_values = unique(data1);

encoded_matrix = zeros(size(data2, 1), length(unique_values));

% 填充 encoded_matrix
for i = 1:size(data2, 1)
    for j = 1:size(data2, 2)
        if ~isnan(data2(i, j))
            % 查找 data2(i, j) 在 unique_values 中的位置
            idx = find(unique_values == data2(i, j));
            % 在 encoded_matrix 中设置对应位置为 1
            encoded_matrix(i, idx) = 1;
        end
    end
end


%编码结束
user_value= encoded_matrix;