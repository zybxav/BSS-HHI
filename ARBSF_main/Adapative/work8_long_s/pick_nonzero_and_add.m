function selected_elements = pick_nonzero_and_add(shuffled_half1)
    % 初始化向量以存储每行选取的非零元素
    selected_elements = zeros(size(shuffled_half1, 1),1);
    
    for i = 1:size(shuffled_half1, 1)
        % 找到当前行中的非零元素，每个用户一行数据，筛选出其中非零的那个
        non_zeros = shuffled_half1(i, shuffled_half1(i, :) ~= 0);
        
        % 如果当前行有非零元素，，用户i就会随机选择一个，randsample
        if ~isempty(non_zeros)
            selected_elements(i) = randsample(non_zeros, 1);
        else
            % 如果该行全为零，可以选择将0作为默认值或其他处理方式
            selected_elements(i) = 0; 
        end
    end
end