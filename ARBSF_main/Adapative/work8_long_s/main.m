dbstop if error
format short
opts = optimoptions('fsolve', 'Display', 'off');
filename1 = 'beta52_input.txt';
k2=13; % 举例，找出频率最高的5个元素
ground_truth_sta
filename = 'data1\beta52_ep1.0_k90.txt';

data_dispove;
pattern = 'ep(\d+\.\d+)';
match = regexp(filename, pattern, 'tokens');
epl = str2double(match{1}{1});

N=length(item);


delay=5;
phase2_pre(d2,item,N,k2,delay,Ground_truth,unique_values,filename,epl);


% delay=10;
%  phase2_pre(d2,item,N,k2,delay,Ground_truth);
% 
% 
% delay=20;
%  phase2_pre(d2,item,N,k2,delay,Ground_truth);
% 
% 
% delay=30;
%  phase2_pre(d2,item,N,k2,delay,Ground_truth);
% 
musicc