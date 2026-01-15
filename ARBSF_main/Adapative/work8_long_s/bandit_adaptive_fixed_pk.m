function result=bandit_adaptive_fixed_pk(d2,k2,item,e,N)
%item_mean=mean(item);
%Pk=item_mean(d2-k+1);
p=e/(e+1);
q=1/(e+1);
item_mean=mean(item);
%这里设定乘以p
Pk=item_mean(d2-k+1)*(p-q)+q;
T=zeros(4,d2);
for o=1:d2;
    T(1,o)=o;
    T(3,o)=1/d2;  %这个是均分的概率
end
%第一行就是value的序号
%第二行就是现有的计算出的次数，要记住所有出现的次数
%第三行是现在的sample概率

%要额外找个排序，找到第m个数据
%最开始一组的应该是统计结果没有0的，或者小于0的位置也设置为1
%先设定第m个参数
%N个用户，每个用户

%这里是应该用了一般的数据来初始化结果
Procedure_matrix=zeros(d2,2);
part1=0.1;
part2=1-part1;
for I=1:ceil(part1*N)
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

%这里我们先暂时设定，误差是扰动后的数据带来的？而不是扰动前的数据
%Pk=pk_test(Procedure_matrix,d2,k2);
    
for jkk=1:d2
    T(3,jkk)=detla_caculate(Procedure_matrix(jkk,1),Procedure_matrix(jkk,2),Pk);
    T(4,jkk)=error_caculate(Procedure_matrix(jkk,1),Procedure_matrix(jkk,2));
end

TH=zeros(1,d2);
tempgogo=0;
for I=ceil(ceil(part1*N))+1:N
    tempgogo=tempgogo+1;
   % Pk=pk_test(Procedure_matrix,d2,k2);
    H=zeros(1,d2);
    TH=T(3,:);
    TH=TH/sum(TH);
    for kk=1:d2
        H(kk,1)=sum(TH(1,1:kk));
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
    if temp0==1
        Procedure_matrix(No,2)=Procedure_matrix(No,2)+1;
    else
        Procedure_matrix(No,1)=Procedure_matrix(No,1)+1;
    end
      %  if rem(tempgogo,d2)==0
    if rem(tempgogo,10)==0
    %Pk=pk_test(Procedure_matrix,d2,k2);
    T(3,No)=detla_caculate(Procedure_matrix(No,1),Procedure_matrix(No,2),Pk,T(3,No));
    end
end

%最后进行数据恢复和统计
Time_sum=Procedure_matrix(:,2)'+Procedure_matrix(:,1)';
Time_sum_1=Procedure_matrix(:,2)';
final_fre=(Time_sum_1-Time_sum*q)./((p-q)*Time_sum);
[~,idx]=sort(final_fre);
T_temp=T(:,idx);
%T(2,:)=final_fre;
result=final_fre;
%然后也要进行排序，找出前面的k个
end