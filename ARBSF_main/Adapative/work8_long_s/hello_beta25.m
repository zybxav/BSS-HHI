
% filename = 'data1\beta52_ep1.0_k90.txt';

pattern = 'ep(\d+\.\d+)';
match = regexp(filename, pattern, 'tokens');
epl = str2double(match{1}{1});

pattern = 'k(\d+)';
match = regexp(filename, pattern, 'tokens');
klk = str2double(match{1}{1});


k2 = klk/3;
delay=5;
phase2_prek(k2,delay,filename,filename1);
%phase2_pre(d2,item,N,k2,delay,Ground_truth,unique_values,filename,epl);


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
