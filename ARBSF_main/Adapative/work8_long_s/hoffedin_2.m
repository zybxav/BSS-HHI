clc;
clear all;
syms delta0 result variance0 theta00 delta0 N t_time
t_time = 1;
% 距离差别应该比较大。就是本身这个数比较小
% result1 = solve(power(log(2/delta0)/(2*t_time),1/2)-theta00==0,delta0);
% delta1 = vpa(1/result1)


%说法是距离越近，概率越高
temp0 = 0;
for theta00 = 0: 0.01:1
  temp0 = temp0 + 1;
  result2 = solve(power(log(2/delta0)/(2*t_time),1/2)-theta00==0,delta0);
  delta2(temp0) = vpa(result2);
end
% 
plot(1:1:temp0, delta2)