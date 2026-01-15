clc;
clear all;
syms delta0 result variance0 theta00 delta0 N t_time


P_sort = 0:0.01:1;
length1=length(P_sort);
% k=5;  %k=5是没有区别，几乎没有改变 k=17变化不大
k=15;
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
T1 = test./sum(test);


% t_time = t_time1;
% temp0 = 0;
% for i = 1:1:length1
%   temp0 = temp0 + 1;
%   if i<=k
%   theta00=abs(P_sort(i)-Pk_1);
%   result2 = solve(power(log(2/delta0)/(2*t_time),1/2)-theta00==0,delta0);
%   else
%        theta00=abs(P_sort(i)-Pk);
%   result2 = solve(power(log(2/delta0)/(2*t_time),1/2)-theta00==0,delta0);
%   end
%   T(temp0) = result2;
% end
% T=T./sum(T);

Num_sum = 10000;
T_number_one_round=T1.*(Num_sum);



for ik= 1:1:10
    Q1=zeros();
    Q2=zeros();
NUB2=1000*ik;
NUB3=Num_sum-NUB2;

T1_number=T1.*NUB2;
% 在T1的基础上迭代得到T2
t_time2 = t_time1+T1_number;
temp0 = 0;
for i = 1:1:length1
  temp0 = temp0 + 1;  if i<=k
  theta00=abs(P_sort(i)-Pk_1);
  result2 = solve(power(log(2/delta0)/(2*t_time2(i)),1/2)-theta00==0,delta0);
  else
       theta00=abs(P_sort(i)-Pk);
  result2 = solve(power(log(2/delta0)/(2*t_time2(i)),1/2)-theta00==0,delta0);
  end
  T2(temp0) =result2;
  % 单纯的把自己代入，概率会是大于1的数据
end
T2 = T2./sum(T2);
T2_number=T2.*NUB3;

T_number= T1_number+T2_number;



% 后面节约的数目
Q1= T_number(TY:length1);
Q2= T_number_one_round(TY:length1);

Q_between=vpa(sum(Q2)-sum(Q1));
H(ik)=Q_between;
end
%有个问题，当nub2=1000时，两者不应该一样吗
plot(1:1:10,H)