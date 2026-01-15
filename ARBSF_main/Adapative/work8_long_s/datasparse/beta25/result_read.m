prefix = 'laplace'; 


v_bandit1=zeros(1,15);
v_bandit2=zeros(1,15);
v_RR=zeros(1,15);


for i=1:1:5
    epsilon=0.5+i*0.5;
    % 将复数 epsilon 转换为字符串，格式为 R+Ii (例如，0.5+0.5i)% 检查 epsilon 是否是整数
if mod(epsilon, 1) == 0
    % 如果 epsilon 是整数
    epsilon_str = sprintf('%d', epsilon);
else
    % 如果 epsilon 是非整数
    epsilon_str = sprintf('%0.1f', epsilon);
end

% 替换文件名中的 '+' 和 'i'，因为文件系统可能不支持这些字符
epsilon_str = strrep(epsilon_str, '+', 'p'); % 用 'p' 替换 '+'
epsilon_str = strrep(epsilon_str, 'i', 'i'); % 如果需要，还可以进一步替换 'i'
filename = sprintf('%s_output.txtEpsilon%s_delay_5k_30.mat', prefix,epsilon_str);
load(filename);
v_bandit1(i)=v11(1);
v_bandit2(i)=v11(2);
v_RR(i)=v11(4);
v_bandit1(i+5)=v22(1);
v_bandit2(i+5)=v22(2);
v_RR(i+5)=v22(4);
v_bandit1(i+10)=v33(1);
v_bandit2(i+10)=v33(2);
v_RR(i+10)=v33(4);
end
v_vector=[v_bandit1;v_bandit2;v_RR]