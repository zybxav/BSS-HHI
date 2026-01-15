dbstop if error

%input
epsilon=1;
e=exp(epsilon);
d2=60;
item=AAA;
N=length(mean(AAA'));
% Ground_truth_p2=mean(AAA);
% Ground_truth_p2=Ground_truth_p2./sum(Ground_truth_p2);


ti=20;
k2=10;
part=5;
v11=zeros;
v22=zeros;
v33=zeros;
v44=zeros;
time_Hu=1;
delay=11;
for Hu=1:1:time_Hu
[v1,v2,v3,v4]=phase2_process(d2,e,item,N,k2,Ground_truth,part,delay);
v11=v11+v1;
v22=v22+v2;
v33=v33+v3;
v44=v44+v4;
end
v=zeros;
v11=v11/time_Hu;
v22=v22/time_Hu;
v33=v33/time_Hu;
v44=v44/time_Hu;
v=[v11,v22,v33,v44]

musicc
%一个是初始数目，携程一个参数的形式
%一个是B2的数目