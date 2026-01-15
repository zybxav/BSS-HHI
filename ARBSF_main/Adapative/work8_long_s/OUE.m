function binary_str = OUE(input_value, epsilon)

p = 1/2;
q = 1 / (exp(epsilon) + 1);

binary_str = zeros(size(input_value));

for i = 1:length(input_value)
    if input_value(i) == 1
        % bit 为 1：以 p 的概率保持 1
        if rand() < p
            binary_str(i) = 1;
        else
            binary_str(i) = 0;
        end
    else
        % bit 为 0：以 q 的概率翻为 1
        if rand() < q
            binary_str(i) = 1;
        else
            binary_str(i) = 0;
        end
    end
end

end
