clear all;
format short
opts = optimoptions('fsolve', 'Display', 'off');
filename = 'Retail_input.txt';
% k2=13; % 举例，找出频率最高的5个元素
delay=5;

%保留10倍更合理
epsilon_vector = [1,1.5,2,2.5,3]
for j=1:5
epsilon=epsilon_vector(j);
k0=300;
k2=30; % 举例，找出频率最高的5个元素
ground_truth_sta
N=num_rows-floor(num_rows/2)+1;
phase2_pre(d2,item,N,k2,delay,Ground_truth,unique_values,filename,epsilon);
end


k2_vector = [10,20,40,50]
for jj=1:4
k2=k2_vector(jj);
epsilon=2;
k0=k2*10;
 % 举例，找出频率最高的5个元素
ground_truth_sta
N=num_rows-floor(num_rows/2)+1;
phase2_pre(d2,item,N,k2,delay,Ground_truth,unique_values,filename,epsilon);
end
