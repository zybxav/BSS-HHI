%这个要写成一个当前
%一共是t轮。
%每一轮的数据我们分为两部分，因为这样数据是已知可靠的
%Dataset_generator;
%计算E1，次数为0.1的数据
p=e/(e+1);
q=1/(e+1);
T=zeros(4,d2);
for o=1:d2;
    T(1,o)=o;
    T(3,o)=1/d2;  %这个是均分的概率
end
tempgogo=0;
Procedure_matrix=zeros(d2,2);

n00=ceil(0.5*N);

for I=1:n00
    tempgogo=tempgogo+1;
    %选定一下No，这里的
    H=zeros();
    for kk=1:d2
        H(kk,1)=sum(T(3,1:kk));
    end
    h_temp=rand(1,1);
    h0=find(H>=h_temp);
    No=min(h0);
    %开始扰动并且记录
    temp1=rand(1,1);
    if  temp1<=p
        temp0= item(I,No);
    else
        temp0=1-item(I,No);
    end
    %如果预估的
    if temp0==1
        Procedure_matrix(No,2)=Procedure_matrix(No,2)+1;
    else
        Procedure_matrix(No,1)=Procedure_matrix(No,1)+1;
    end
end

%第一步
%对应的错误概率和是
Pmatrix=sum(Procedure_matrix');
s=zeros();
for i=1:d
  s(i)=power(log(10)/(2*Pmatrix(i)),1/2);
end
%The error distance
s0=sum(s);

%I need to reallocate their assigment
Pk=pk_test(Procedure_matrix,d2,k2);
Time=zeros();
for jkk=1:d
    Time(jkk)=detla_caculate(Procedure_matrix(jkk,1),Procedure_matrix(jkk,2),Pk);
end


%具体的设定如下：每一轮都两部分
%我们的目标函数是：  设定数据分配和对应误差的乘积，代表了 错误分配的指标，希望这个指标是越小越好   
%第0部分肯定有个基础内容，这是一个和数据相关的分布
%第1部分的数目是n1, 对应的误差是  E1，这个是和数据相关，和数据的次数有关，用hoeffding  第0部分得出的误差
%第2部分的数目是n-n1,  对应的误差是 E2 第1部分的误差。
%第2部分的数据再多也没啥意义，因为其减小的误差

%和收集结果相关，假定variance不变可以做一个。

%one-round, error decreaing
%no need to run the second round
%caculate E2


syms n11
syms n22
Pmatrix2=Pmatrix+n11*Time;
n22=N-n00-n11;
syms s11
for i=1:d
  s11(i)=power(log(10)/(2*Pmatrix2(i)),1/2);
end
%The error distance
s1=sum(s11);


%this item is a 
Y=n11*s0+n22*s1;
%条件1：其中E2和n1相关
%条件2：其中E1+E2为定值


dy=diff(Y);
n11=solve(dy);
%要清楚一个时间
%再带回去呢

 dy=vpa(subs(Y, n11)) 
 %结果是s0=2.88, s1=1.24
  n22=vpa(subs(n22, n11))
 

