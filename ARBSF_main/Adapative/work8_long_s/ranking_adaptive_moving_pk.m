function [result,Time_sum,n22]=ranking_adaptive_moving_pk(d2,k2,item,e,N,part1,delay)
%item_mean=mean(item);
%Pk=item_mean(d2-k+1);
n22=zeros();
p=e/(e+1);
q=1/(e+1);
T=zeros(4,d2);
T(1, :) = 1:d2; % 将序号 1 到 d2 赋值给第一行
T(3, :) = 1 / d2; % 将 1/d2 赋值给第三行
%第一行就是value的序号
%第二行就是现有的计算出的次数，要记住所有出现的次数
%第三行是现在的sample概率

%要额外找个排序，找到第m个数据
%最开始一组的应该是统计结果没有0的，或者小于0的位置也设置为1
%先设定第m个参数
%N个用户，每个用户

%这里是应该用了一般的数据来初始化结果
Procedure_matrix=zeros(d2,2);
part2=part1*d2+1;
for I=1:ceil(part1*d2)
    No=rem(I,d2)+1;
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
if part2<N
%这里我们先暂时设定，误差是扰动后的数据带来的？而不是扰动前的数据
[Pk,Pk1,pika,theta00]=pk_test(Procedure_matrix,d2,k2);
TT=Procedure_matrix(:,2)'./(Procedure_matrix(:,1)+Procedure_matrix(:,2))';

for jkk=1:d2
    if TT(jkk)>=Pk
        T(3,jkk)=detla_caculate(Procedure_matrix(jkk,1),Procedure_matrix(jkk,2),Pk);
    else
        T(3,jkk)=detla_caculate(Procedure_matrix(jkk,1),Procedure_matrix(jkk,2),Pk1);
    end
    T(4,jkk)=detla_caculate_fre(Procedure_matrix(jkk,1),Procedure_matrix(jkk,2),T(4,jkk),theta00);
end
n22=zeros();
n11=ceil(n1i_caculation(Procedure_matrix,N-I,d2,k2)) ;
n22(1)=n11;
TH=zeros(1,d2);
tempgogo=0;
record=zeros(1,d2);
delay_test=1;
for I=ceil(part2)+1:N
    tempgogo=tempgogo+1;
    [Pk,Pk_1,pika,theta00]=pk_test(Procedure_matrix,d2,k2);
    TH2=zeros();
    TH3=zeros();
    TH=T(3,:);
    TH2=10*pika.*T(4,:);
    TH3=TH2+TH;
    TH4=TH3/sum(TH3);
    H=zeros(1,d2);
    
    for kk=1:d2
        H(kk,1)=sum(TH4(1,1:kk));
    end
    h_temp=rand(1,1);
    h0=find(H>=h_temp);
    No=min(h0);
    record(No)=record(No)+1;
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
    % if rem(tempgogo,600)==0
        if (delay_test<=delay)
    if (tempgogo==ceil(part2+n11))
            delay_test
              delay_test=delay_test+1;
        %为什么只有这个地方更新，应该全部更新才对
        TT=Procedure_matrix(:,2)'./(Procedure_matrix(:,1)+Procedure_matrix(:,2))';
        [Pk,Pk1,pika,theta00]=pk_test(Procedure_matrix,d2,k2);
        for jkk=1:d2
            if record(jkk)>=1
                if TT(jkk)>=Pk1
                    T(3,jkk)=detla_caculate(Procedure_matrix(jkk,1),Procedure_matrix(jkk,2),Pk);
                else
                    T(3,jkk)=detla_caculate(Procedure_matrix(jkk,1),Procedure_matrix(jkk,2),Pk1);
                end
                T(4,jkk)=detla_caculate_fre(Procedure_matrix(jkk,1),Procedure_matrix(jkk,2),T(4,jkk),theta00);
            end
        end  %应该是全部更新一次
        n11=n11+ceil(n1i_caculation(Procedure_matrix,N-I,d2,k2));
 n22=[n22;n11];
    end
        end
 record=zeros(1,d2);
end
end
%最后进行数据恢复和统计
Time_sum=Procedure_matrix(:,2)'+Procedure_matrix(:,1)';
Time_sum_1=Procedure_matrix(:,2)';
%temptemp = sum(Procedure_matrix');
final_fre=N./Time_sum.*(Time_sum_1-Time_sum*q)./((p-q));
[~,idx]=sort(final_fre);
T_temp=T(:,idx);
%T(2,:)=final_fre;
result=final_fre;
%然后也要进行排序，找出前面的k个

end