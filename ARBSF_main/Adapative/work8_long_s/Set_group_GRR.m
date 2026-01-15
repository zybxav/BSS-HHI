%返回set中的任意一个值
epsilon=1;
e=exp(1);
%GRR版本
%共d个维度
item=data;
ti=10;

%返回set中的任意一个值
%解决一下最后参数的事情
k=30;
vector_SRR=SRR(d,e,item,N,ti);
vector_GRR=GRR(d,e,item,N);
vector_OLH=OLH(d,e,item,N);
[result_SRR,xor_SRR]=ncr(N,k,d,Ground_truth,vector_SRR)
[result_GRR,xor_GRR]=ncr(N,k,d,Ground_truth,vector_GRR)
[result_OLH,xor_OLH]=ncr(N,k,d,Ground_truth,vector_OLH);
