%n_l=1/((power(theta0,2)/(2*log(m/delta0))-delta0/3)*(power(2*p-1,2)/(p*(1-p))))
%我们应该使用求出对应的t
epsilon=0;
p=exp(epsilon)/(exp(epsilon)+1);
variance=power(p*(1-p),1/2);
R=1;
 t=10000;
 theta1=R*power(log(2/delta)/(2*t),1/2)
 theta2=variance*(power(2*log(3/delta)/t,1/2))+3*R*log(3/delta)/t
% theta=0.0001;
% delta=0.0001;
% t1=log(2/delta)/(2*power(theta/R,2))
% t2=