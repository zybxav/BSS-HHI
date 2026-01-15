function [result,Time_sum]=RR(d2,item,e,N)
%item_mean=mean(item);
%Pk=item_mean(d2-k+1);
p=e/(e+1);
q=1/(e+1);
T=zeros(4,d2);

Procedure_matrix=zeros(d2,2);
for I=1:N
     No=randsample(1:d2,1);
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

%最后进行数据恢复和统计
Time_sum=Procedure_matrix(:,2)'+Procedure_matrix(:,1)';
Time_sum_1=Procedure_matrix(:,2)';
final_fre=d2.* (Time_sum_1-(N/d2)*q)./((p-q));
result=final_fre;
% save T4
end