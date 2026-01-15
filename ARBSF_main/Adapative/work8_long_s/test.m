dbstop if error
%返回set中的任意一个值
d2=60;
item=data(:,d-d2+1:d);
Ground_truth2=mean(item);
 %这是个啥
epsilon=10;
e=exp(epsilon);

%返回set中的任意一个值
%解决一下最后参数的事情
k2=30;
%vector_SRR=SRR(d,e,item,N,ti);
vector_bandit_moving_pk=ranking_adaptive_moving_pk(d2,k2,item,e,N);