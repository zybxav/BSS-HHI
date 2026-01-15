%dbstop if error
%data_generate_designi
%d2=d;
% 文件名


% 读取文件到变量data中
% 假设数据是用空格或逗号分隔的，需要根据实际情况选择适合的函数和参数
% data = readmatrix(filename, 'Delimiter', ',');
% if any(isnan(data(:,1)))
%     data = readmatrix(filename, 'Delimiter', ' ');
% end
data = shuffled_half_final;
data(isnan(data)) = 0;
data2 = data;%备份
data1 = data(~isnan(data));
data1 = data1(data ~= 0);%提取出所有的item，用来统计true_distribution

% 使用unique函数找出所有不同的数值
unique_values = unique(data1);
d2=length(unique_values);   
encoded_matrix = zeros(size(data2, 1), length(unique_values));%创建全0矩阵
%矩阵每行代表一个用户，列数均为d2
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
item= encoded_matrix;