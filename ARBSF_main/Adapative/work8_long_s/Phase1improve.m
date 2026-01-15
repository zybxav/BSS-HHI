load('POS.mat')
epsilon=3;
e=exp(epsilon);
k=3;
d=100;
p_com=e/(k*e+d-k);
q_com=1/(k*e+d-k);
q_com_sum=(d-k)*q_com;
P1=k*p_com
p_krr=e/(e+d-1);
q_krr=1/(e+d-1);
q_krr_sum=(d-1)*q_com;
P2=p_krr
%回去确认一下第一阶段，以及看一下能否推广到QL中。
%其实是因为sample的不均匀，爱是和对方的方法一样，其实是很好用的。
%其实作为一个小改进，完全没问题
%sample 到自己组里的概率是
p_com2=P1;
q_com2=1-P1;
T1=q_com2*(1-q_com2)/power(p_com2-q_com2,2)
%分配对的概率
p_krr2=p_krr+(k-1)*q_krr;
q_krr2=1-p_krr2;
T2=q_krr2*(1-q_krr2)/power(p_krr2-q_krr2,2)

%???这个方差代表了什么？集合内扰动到集合内的概率更大，扰动到集合外的概率不是很小嘛？
%方差大，不代表准

% T3=q_com*(1-q_com)/power(p_com-q_com,2)
% %分配对的概率
 %T4=q_krr*(1-q_krr)/power(p_krr-q_krr,2)


%好奇怪，这个varicance和espilon无关