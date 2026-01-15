clc;
clear all;
syms delta0 Num_sum result2
global delta0 result2

Num_sum = 10000;
P_sort = 0:0.025:1;
length1=length(P_sort);
% k=5; 


%variance是0的时候收敛最快，但是后面variance越来越大
k=30;
Pk=P_sort(k);
Pk_1=P_sort(k+1);
TY=min(find(P_sort>Pk_1));
H =zeros();
temp0 = 0;
test= zeros;
T1_time = 1000;
%当variance收敛到0，就该更新了 
variance0=0.3;
for i = 1:1:length1
  temp0 = temp0 + 1;
  if i<=k
  theta00=abs(P_sort(i)-Pk_1);
  result2 = solve((power(variance0*2*log(3/delta0)/T1_time,1/2))+3*1*log(3/delta0)/T1_time-theta00==0,delta0);
  else
  theta00=abs(P_sort(i)-Pk);
  result2 = solve((power(variance0*2*log(3/delta0)/T1_time,1/2))+3*1*log(3/delta0)/T1_time-theta00==0,delta0);
  end
  test(temp0)= result2;
end
T1=test;
L=50;

dr2=zeros(1,L);
dr3=zeros(1,L);
delta2 =zeros(1,L);
delta3 = zeros(1,L);
T_number_one_round=T1*(Num_sum);
inlin=1;
HL=1;
for t=1:1:L
T2_number=50.*t;
t_time2 =zeros(1,length1)+T1_time+T2_number;
test22=caculate_variation2(length1,P_sort, Pk_1, Pk, t_time2,k,variance0);
%归一化
t_time3 = zeros(1,length1)+T1_time+T2_number-HL;
test33=caculate_variation2(length1,P_sort, Pk_1, Pk, t_time3,k,variance0);
Test33=test33/sum(test33);

%更新后的比例是
t_time4=t_time3+Test33*HL; 
test44=caculate_variation2(length1,P_sort, Pk_1, Pk, t_time4,k,variance0);
    test22(find(test22>=1))=1;
   test44(find(test44>=1))=1;

% delta2(t)=sum(test22);
% delta3(t)=sum(test44);
delta2(t)=vpa(1-prod(1-test22));
 delta3(t)=vpa(1-prod(1-test44));
end

Tgaga=delta3-delta2
%就一直没有更新吗
bar(Tgaga)

%因为本身的数目减小最多
%那就是说直接找最多的那一刻就停止就好了

bar(Tgaga)

