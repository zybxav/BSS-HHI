function test=caculate_variation2(length1,P_sort, Pk_1, Pk, t_time,k,variance0)
%第2阶段结束后计算数量
global delta0 
test=zeros();
for i = 1:1:length1
  if i<=k
  theta00=abs(P_sort(i)-Pk_1);
  result2 = solve((power(variance0*2*log(3/delta0)/t_time(i),1/2))+3*1*log(3/delta0)/t_time(i)-theta00==0,delta0);
  else
  theta00=abs(P_sort(i)-Pk);
  result2 = solve((power(variance0*2*log(3/delta0)/t_time(i),1/2))+3*1*log(3/delta0)/t_time(i)-theta00==0,delta0);
  end
  test(i) =result2;
end
end