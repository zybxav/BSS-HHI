function [ncr, var, accu,f1,mcr] = sort_compare2(idx_top, nums_top, ground_top, nums1_top, k, N)
%idx_top为预测的topk items,nums_top为预测的frequency，gt和nums1_top则是对应的真实topk items

% Initialize the score array
    arr_score = k:-1:1;
    arr_score_sum = sum(arr_score);
    
    % Convert idx_top and ground_top to sets
    a = unique(idx_top);
    b = unique(ground_top);
    
    % Find common elements between a and b
    c = intersect(a, b);
    
    % Calculate accuracy
    accu = length(c) / k;
    
    % Calculate normalized cumulative rank (ncr)
    score = 0;
    for i = 1:length(c)
        result = find(ground_top == c(i), 1); % Find the first occurrence of c(i) in ground_top
        score = score + arr_score(result);
    end
    ncr = score / arr_score_sum;
    
    % Calculate variance
    var = 0;
    for j = 1:length(b)
        if any(a == b(j))
            result = find(ground_top == b(j), 1); % Find the first occurrence of b(j) in ground_top
            var = var + (nums_top(result) - nums1_top(result))^2;
        else
            result = find(ground_top == b(j), 1); % Find the first occurrence of b(j) in ground_top
            var = var + (nums1_top(result) - 0)^2;
        end
    end
    var = var / k;
    var = var / N;

    % Return the calculated metrics

    %calculate_f1_score
    precision = accu;
    recall = accu;

    if precision + recall > 0
        f1 = 2 * precision * recall / (precision + recall);
    else
        f1 = 0;
    end
    %calculate_mcr
    hit_sum = 0;
    total_sum = sum(nums1_top);   % sum of all true frequencies (denominator)
    
    for i = 1:length(c)
        result = find(ground_top == c(i), 1);
        hit_sum = hit_sum + nums1_top(result);
    end
    
    if total_sum > 0
        mcr = double(hit_sum) / double(total_sum);
    else
        mcr = 0;
    end
end
