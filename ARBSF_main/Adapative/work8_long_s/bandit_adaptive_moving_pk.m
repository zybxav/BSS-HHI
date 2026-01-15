function [result,Time_sum,n22]=bandit_adaptive_moving_pk(d2,k2,item,e,N,part1,delay)
%item_mean=mean(item);
%Pk=item_mean(d2-k+1);
n22=zeros();
p=e/(e+1);
q=1/(e+1);
T=zeros(4,d2);

T(1, :) = 1:d2; % 将序号 1 到 d2 赋值给第一行
T(3, :) = 1 / d2; % 将 1/d2 赋值给第三行

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
    [Pk,Pk1,pika,~]=pk_test(Procedure_matrix,d2,k2);
    TT=Procedure_matrix(:,2)'./(Procedure_matrix(:,1)+Procedure_matrix(:,2))';
    
    parfor jkk=1:d2
        if TT(jkk)>=Pk
            T(3,jkk)=detla_caculate(Procedure_matrix(jkk,1),Procedure_matrix(jkk,2),Pk);
        else
            T(3,jkk)=detla_caculate(Procedure_matrix(jkk,1),Procedure_matrix(jkk,2),Pk1);
        end
    end
    
    n11=ceil(n1i_caculation(Procedure_matrix,N-I,d2,k2));
    n22(1)=n11
    TH=zeros(1,d2);
    tempgogo=0;
    record=zeros(1,d2);
    delay_test=1;
    for I=ceil(part2)+1:N
        tempgogo=tempgogo+1;
        Pk=pk_test(Procedure_matrix,d2,k2);
        H=zeros(1,d2);
        TH=T(3,:);
        TH=TH/sum(TH);
        for kk=1:d2
            H(kk,1)=sum(TH(1,1:kk));
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
        
        % if rem(tempgogo,n11)==0
        if (delay_test<=delay)
            
            
            if (tempgogo==ceil(part2+n11))
                delay_test
                delay_test=delay_test+1;
                TT=Procedure_matrix(:,2)'./(Procedure_matrix(:,1)+Procedure_matrix(:,2))';
                [Pk,Pk1,pika,~]=pk_test(Procedure_matrix,d2,k2);
                for jkk=1:d2
                    if record(jkk)>=1
                        if TT(jkk)>=Pk1
                            T(3,jkk)=detla_caculate(Procedure_matrix(jkk,1),Procedure_matrix(jkk,2),Pk);
                        else
                            T(3,jkk)=detla_caculate(Procedure_matrix(jkk,1),Procedure_matrix(jkk,2),Pk1);
                        end
                    end
                end  %应该是全部更新一次
                % bar(T(3,:));
                n11=n11+ceil(n1i_caculation(Procedure_matrix,N-I,d2,k2));
                n22=[n22,n11];
            end
        end
        record=zeros(1,d2);
    end
end
%最后进行数据恢复和统计
Time_sum=Procedure_matrix(:,2)'+Procedure_matrix(:,1)';
Time_sum_1=Procedure_matrix(:,2)';
%temptemp = sum(Procedure_matrix');
%恢复算法里这是错的
final_fre=N./Time_sum.*(Time_sum_1-Time_sum*q)./((p-q));
result=final_fre;
end