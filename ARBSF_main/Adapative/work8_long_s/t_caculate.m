function T2=t_caculate(k, P_sort,Pk, Pk_1,t_time,length1)

temp0=0;
for i = 1:1:length1
      syms  delta0
    result=0;
  temp0 = temp0 + 1; 
  if i<=k
  theta00=abs(P_sort(i)-Pk_1);
  result2 = solve(power(log(2/delta0)/(2*t_time(i)),1/2)-theta00==0,delta0)
  else
  theta00=abs(P_sort(i)-Pk);
  result2 = solve(power(log(2/delta0)/(2*t_time(i)),1/2)-theta00==0,delta0)
  end
  T2(temp0) =result2;
  clear delta0
end
end