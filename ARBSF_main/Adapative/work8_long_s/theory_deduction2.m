clc;
clear all;
syms NUB2 NUB3 delta0
NUB2=5000;
user_number=10000;
NUB3=user_number-NUB2 
P_sort = 0:0.01:1;
length1=length(P_sort);

k=5;
Pk=P_sort(k);
Pk_1=P_sort(k+1);
TY=min(find(P_sort>Pk_1));
H =zeros();
%one-round scheme
%%%%%%%%%%%%%%%%一次性分配1000次
temp0 = 0;
test= zeros;
t_time1 = 100;
for i = 1:1:length1
  temp0 = temp0 + 1;
  if i<=k
  theta00=abs(P_sort(i)-Pk_1);
  result2 = solve(power(log(2/delta0)/(2*t_time1),1/2)-theta00==0,delta0);
  else
       theta00=abs(P_sort(i)-Pk);
  result2 = solve(power(log(2/delta0)/(2*t_time1),1/2)-theta00==0,delta0);
  end
  test(temp0)= result2;
end
T11=test;
T1 = test./sum(test);

T_number_one_round=T1*(user_number);
T1_number=T1.*NUB2;

% 在T1的基础上迭代得到T2
t_time2 = t_time1+T1_number;
temp0 = 0;
for i = 1:1:length1
  temp0 = temp0 + 1; 
  if i<=k
  theta00=abs(P_sort(i)-Pk);
  result2 = solve(power(log(2/delta0)/(2*t_time2(i)),1/2)-theta00==0,delta0);
  else
       theta00=abs(P_sort(i)-Pk);
  result2 = solve(power(log(2/delta0)/(2*t_time2(i)),1/2)-theta00==0,delta0);
  end
  T2(temp0) =result2;
  % 单纯的把自己代入，概率会是大于1的数据
end
T22=T2;
T2 = T2./sum(T2);
T2_number=T2.*NUB3;
T_number= T1_number+T2_number;

%定义为复杂度。
delta1=sum(T11);
delta2=sum(T22);


%delta1=vpa(1-prod(1-T11))
%delta2=vpa(1-prod(1-T22)) 


% y1 = diff(delta2,NUB2);
% NUB2=vpasolve(y1==0)
% eval(NUB2)

% 不断在更正数据，不需要的数据比例就比较少
