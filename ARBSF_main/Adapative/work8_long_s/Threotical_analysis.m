clc;
clear all;
%我们写频率
%var=1/n*((p*(1-p))\power(p-q,2)+)
%n个属性
 syms p_oue q_oue p_krr q_krr k N 
 N=900000;
 n=300000;
 k=64;
epsilon=1;
e=exp(epsilon);
 p_oue=e/(e+1);
 q_oue=1/(e+1);
 

 s1=0.4;   %百分之10的数据用来测数目了
 s2=0.6;
%Var_oue=power(l1,2)*(n1*q_oue*(1-q_oue))/power(p_oue-q_oue,2);
% Var_krr=power(l2,2)*(n2*q_krr*(1-q_krr))/power(p_krr-q_krr,2);
%baseline，方差和k无关
l1=n;
n1=N/n;
Var_base=(1/n1*q_oue*(1-q_oue))/power(p_oue-q_oue,2)

%LDPminer1
%第一阶段,N/2,n1=N/2/n，l1=n
l1=n;
n1=N*s1/n;
Var_rr_miner=(1/n1*q_oue*(1-q_oue))/power(p_oue-q_oue,2)
Var_olh_miner=1/n1*(4*1^2*e)/power(e-1,2)
%第二阶段,n2=N/2,l2=2*k
l2=2*k;
n2=N*s2;
p_krr=e/(e+l2-1);
q_krr=1/(e+l2-1);
Var_krr_miner1=power(l2,2)*(1/n2*q_krr*(1-q_krr))/power(p_krr-q_krr,2)
Var_OLH1=1/n2*(4*l2^2*e)/power(e-1,2)
%缩小一部分数据，假定k变小了。
%LDPminer2
%第二阶段,n2=N/2,l2=2*k
l3=k;
n2=N*s2;
p_krr=e/(e+l3-1);
q_krr=1/(e+l3-1);
Var_krr_miner2=power(l3,2)*(1/n2*q_krr*(1-q_krr))/power(p_krr-q_krr,2)
Var_OLH2=1/n2*(4*l3^2*e)/power(e-1,2)


%2阶段OUE算法
l1=n;
n1=N*s1/n;
Var_oue_miner21=(1/n1*q_oue*(1-q_oue))/power(p_oue-q_oue,2)
l2=2*k;
n2=N*s2/l2;
Var_rr_miner22=(1/n2*q_oue*(1-q_oue))/power(p_oue-q_oue,2)
Var_OLH2=1/n2*(4*12^2*e)/power(e-1,2)


%？？？我现在跑实验出来，结果不对啊。
%上述这个算法其实第一轮找2k根本没有保证正确性，没有指标判断

H1=ceil(log2(n/k));
H1_vector=zeros(1,H1);

for iH0=1:H1
H1_vector(iH0)=power(iH0,1)*1/H1;
end
H1_vector=H1_vector/sum(H1_vector);
l4=2*n;
n44=0;
for iH1=1:H1
    l4=l4/2;
    n4=N*H1_vector(iH1)/l4;
    n44=n44+n4;
    Var_rr_halving=(1/n4*q_oue*(1-q_oue))/power(p_oue-q_oue,2);
end
 Var_rr_halving_multi=(1/n44*q_oue*(1-q_oue))/power(p_oue-q_oue,2)

 %全部的测试
 l5=2*k;
 n55=N/l5;
  Var_rr_test=(1/n55*q_oue*(1-q_oue))/power(p_oue-q_oue,2)
 
 
% K1=Var_base;
% K2=Var_krr_miner1;   
% K3=Var_krr_miner2; 
% K4=Var_oue_miner22;
% K5=Var_oue_halving;
%K=[K1;K2;K3;K4;K5]
K1=Var_base;
K2=Var_OLH1;   
K3=Var_OLH2; 
K4=Var_rr_miner22;
K5=Var_rr_halving_multi;
K6=Var_rr_test;
KK=[K1;K2;K3;K4;K5;K6]

%疑惑。。。对方的比较对象是啥？真的是人类疑惑
%按照精心的设计，怎么都可以比对方强。譬如以k为一个最优情况？不会比k再好了吧？