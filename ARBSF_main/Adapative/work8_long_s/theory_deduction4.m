clc;
clear all;
syms delta0 Num_sum

Num_sum = 200;
%分多轮就有多个参数

P_sort = 0:0.02:1;
length1=length(P_sort);
% k=5;  %k=5是没有区别，几乎没有改变 k=17变化不大
k=20;
Pk=P_sort(k);
Pk_1=P_sort(k+1);
TY=min(find(P_sort>Pk_1));
H =zeros();
%one-round scheme
%%%%%%%%%%%%%%%%一次性分配1000次
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
%T1 = test./sum(test);
T1=test;


T_number_one_round=T1*(Num_sum);

T2_number=T1.*Num_sum;

% 在T1的基础上迭代得到T2，这个相当于在获取第二阶段的值以后跑第3阶段
t_time2 = T1_time+T2_number;
%第2阶段结束后计算数量
temp0 = 0;
%其实是第2阶段结束后的参数
test2=zeros();
for i = 1:1:length1
  temp0 = temp0 + 1; 
  if i<=k
  theta00=abs(P_sort(i)-Pk_1);
  result2 = solve(power(log(2/delta0)/(2*t_time2(i)),1/2)-theta00==0,delta0);
  else
       theta00=abs(P_sort(i)-Pk);
  result2 = solve(power(log(2/delta0)/(2*t_time2(i)),1/2)-theta00==0,delta0);
  end
  test2(temp0) =result2;
end

%T2 = test2./sum(test2);
T2=test2
delta1=vpa(1-prod(1-T1))
delta2=vpa(1-prod(1-T2))

%这里应该是可以求出其对应的  变化率，开头到结尾分成几份，将对应的份额求出来。
%只把第一部分的求出来。
r=10;
%找到其对应的变化位置
delta_target = delta1+1/r*abs(delta1-delta2)








