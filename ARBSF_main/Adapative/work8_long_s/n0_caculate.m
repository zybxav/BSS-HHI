function h=n0_caculate(N,d,epsilon)
syms  h theta
%这里的theta是啥,默认是1吗？
%为什么和epsilon无关，我记得是越小关系越大拉着
e=exp(epsilon);
p=e/(1+e);
q=1-p;
theta=p-q;
T=(N-h*d)*(1-d*exp(-power(theta,2)*h/(2*1)));
dy=diff(T,h);
h=solve(dy);
h=ceil(vpa(h));
%每一轮选择最小的
end
%根据总样本数 N、item 数 d、隐私预算 ε，利用概率和 Hoeffding 边界分析公式，计算每个 item 的最优抽样数量 h，保证 LDP Top-k 估计的准确性。