%写一个LDPminer的
% 40 10 50
%按照当前的分配来做。
%第1阶段先找2k
%第2阶段找数目
%最后一个阶段，范围内扰动
clc;
clear all;
n = 40;   % 设置机器的个数
m=10;  %这是个啥
N=5000;
Dataset_generator

% 设置参数的值

t=zeros(2,n);
t(1,:)=0:1/(n):1-1/n;
%这个实验的流程是，我们先不用用户的数据做，应该是没办法回复的吧
%t(2,:)=zeros(1,n)+1;
epsilon=1;
e=exp(epsilon);
p=e/(e+1);
q=1/(e+1);


T=zeros(3,n);
for o=1:n;
    T(1,o)=o;
    T(3,o)=1/n;  %这个是均分的概率
end
    for k=1:n
         H(k,1)=sum(T(3,1:k));
     end
T1=zeros(2,n);
Uo=0;
Procedure_matrix=zeros(n,2);
for i1=1:ceil(0.4*N)
     Uo=Uo+1;
     h_temp=rand(1,1);
     h0=find(H>=h_temp);
     No=min(h0);
      %开始扰动并且记录
            temp1=rand(1,1);
            if  temp1<=p
                  temp0= data(Uo,No);
            else
                   temp0=1-data(Uo,No);
            end 
            if temp0==1
                  Procedure_matrix(No,2)=Procedure_matrix(No,2)+1;
            else
                  Procedure_matrix(No,1)=Procedure_matrix(No,1)+1;
            end  
end
%%扰动找出2k
%首先要排序
T(2,:)=Procedure_matrix(:,2)./(Procedure_matrix(:,2)+Procedure_matrix(:,1));
  [~,idx]=sort(T(2,:));
  T_temp=T(:,idx);
  T_sort1=T(1,1:n-2*m);
T_sort2=T(1,n-2*m+1:n);
Data2=data;
for i11=1:n-2*m
    Data2(T_sort(1,i11),:)=0;
end
%将Data2对应地方置为0

%后续的
k_T2=zeros(1,2*m);
k_temp=0;
for i2=Uo:ceil(0.1*N)
%探测这10%的用户
Uo=Uo+1;
%这边也是KRR，最大是2k，现在的算法也很恶心
k0=sum(Data2(Uo,:));
     if k0>k_temp
         k_temp=k0
     end
end

epsilon=1;
e=exp(epsilon);
p3=e/(e+k-1);
q3=1/(e+k-1);
pq=zeros(1,k_temp);
pq(1)=p3;
for i22=2:k_temp
   pq(i22)=pq(i22-1)+q3;
end
%第3阶段的KRR
vector3=zeros(1,2*k);
for i3=Uo:N
    %所有的长度是  k_temp;
    k0=sum(Data2(Uo,:));
    table=find(Data2(Uo,:)==1);
    %将非0的部分和其位置找出来，做成一个向量
    if K0<k_temp
        temp_K0=k_temp-K0;
        temp_table=zeros(temp_K0,1);
       table=[table;temp_table];
    end
    %选择里面的非1的数目扰动，KRR的扰动数目是k_temp+1
   key_rand=round(rand(1,1)*(k_temp))+1; 
   %随意选择一个数，然后上传
  key_temp=table(key_rand);
    %KRR扰动
   %生成其扰动到的对象
  rand_number=rand(1,1);
  HH1=find(pq<=rand_number);
  HH2=max(HH1);
  %输出的对象
  out0=table(HH2);
  vector3(out0)=vector3(out0)+1;
end
n0=N-U0+1;
%这个n0不对劲
 vector4=(vector3 - n0*q3)/(n0*p3 - n0*q3)

%KRR的数据恢复咋写

