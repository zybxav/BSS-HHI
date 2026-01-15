clc;
clear all;
syms delta0 Num_sum
global delta0

Num_sum = 10000;
%分多轮就有多个参数
P_sort = 0:0.01:1;
length1=length(P_sort);
% k=5;  %k=5是没有区别，几乎没有改变 k=17变化不大
k=30;
Pk=P_sort(k);
Pk_1=P_sort(k+1);
TY=min(find(P_sort>Pk_1));
H =zeros();
temp0 = 0;
test= zeros;
T1_time = 100;
for i = 1:1:length1
  temp0 = temp0 + 1;
  if i<=k
  theta00=abs(P_sort(i)-Pk_1);
  result2 = solve(power(log(2/delta0)/(2*T1_time),1/2)-theta00==0,delta0);
  else
       theta00=abs(P_sort(i)-Pk);
  result2 = solve(power(log(2/delta0)/(2*T1_time),1/2)-theta00==0,delta0);
  end
  test(temp0)= result2;
end
T1=test;
L=50;

dr2=zeros(1,L);
dr3=zeros(1,L);
delta2 =zeros(1,L);
delta3 = zeros(1,L);
T_number_one_round=T1*(Num_sum);
inlin=200;

for t=1:1:L
T2_number=T1.*inlin.*t;
t_time2 = T1_time+T2_number;
test2=caculate_variation(length1,P_sort, Pk_1, Pk, t_time2,k);
%归一化
if t>=2
Test2=test2/sum(test2);
T3_number= Test2.*inlin;
t_time3=T1_time+T1.*inlin.*(t-1)+T3_number; 
test3=caculate_variation(length1,P_sort, Pk_1, Pk, t_time3,k);
end
% dr2(t)=test2;
% dr3(t)=test3;
delta2(t)=sum(test2);
if t==1
   delta3(1)=0; 
else
   delta3(t)=sum(test3);
end
end
delta2(2:L)-delta3(2:L)
%一开始前者误差比较大，后面后者大，这时候应该换了。


%这里应该是可以求出其对应的  变化率，开头到结尾分成几份，将对应的份额求出来。
% %只把第一部分的求出来。
% bar(dr)
% x=200:200:10000
% plot(x,dr)




