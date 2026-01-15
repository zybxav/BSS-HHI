function shuffled_half_final = process_matrix(shuffled_half2, indices)
    % 第1步：保留indices中存在的数据，其余置为0
    mask = ismember(shuffled_half2, indices);
    shuffled_half2(~mask) = 0;
    
    % 第2步：每行将非0元素放在前面，实现非零元素前移
    for i = 1:size(shuffled_half2, 1)
        non_zeros = shuffled_half2(i, shuffled_half2(i, :) ~= 0);  % 提取非零元素
        zeros_count = sum(shuffled_half2(i, :) == 0);  % 计算零元素的数量
        shuffled_half2(i, :) = [non_zeros, zeros(1, zeros_count)];  % 重新排列行
    end
    
    % 第3步：删除全都是0的列
    col_sum = sum(shuffled_half2, 1);  % 计算每列的和
    shuffled_half2(:, col_sum == 0) = [];  % 删除全零列，此处由2497列缩减到了136列
    
    % 返回处理后的矩阵
    shuffled_half_final = shuffled_half2;
end