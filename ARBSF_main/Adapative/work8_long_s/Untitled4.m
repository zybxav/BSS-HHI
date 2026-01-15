clc;
clear all;
syms delta0 result variance0 theta00 delta0 N t_time
P_sort = 0:0.05:1;
k=10;
Pk=P_sort(k);
Pk_1=P_sort(k+1);
temp0 = 0;
t_time = 100
T1=zeros();
for i = 1:1:21
  temp0 = temp0 + 1;
  if i<=k
  theta00=abs(P_sort(i)-Pk_1);
  result2 = solve(power(log(2/delta0)/(2*t_time),1/2)-theta00==0,delta0);
  else
       theta00=abs(P_sort(i)-Pk);
  result2 = solve(power(log(2/delta0)/(2*t_time),1/2)-theta00==0,delta0);
  end
  T1(temp0) = result2;
end
bar(T1)