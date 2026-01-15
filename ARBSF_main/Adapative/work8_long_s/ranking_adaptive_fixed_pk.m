function result=ranking_adaptive_fixed_pk(d2,k2,item,e,N)

p=e/(e+1);
q=1/(e+1);
item_mean=mean(item);

Pk=item_mean(d2-k+1)*(p-q)+q;
T=zeros(4,d2);
for o=1:d2;
    T(1,o)=o;
    T(3,o)=1/d2;  
end

Procedure_matrix=zeros(d2,2);
part1=0.1;
part2=1-part1;
for I=1:ceil(part1*N)

    H=zeros();
    for kk=1:d2
        H(kk,1)=sum(T(3,1:kk));
    end
    h_temp=rand(1,1);
    h0=find(H>=h_temp);
    No=min(h0);

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
end

Pk=pk_test(Procedure_matrix,d2,k2);
    
for jkk=1:d2
    T(3,jkk)=detla_caculate(Procedure_matrix(jkk,1),Procedure_matrix(jkk,2),Pk,T(3,jkk));
     %T(4,jkk)=variance_caculate_fre(Procedure_matrix(jkk,1),Procedure_matrix(jkk,2));
     T(4,jkk)=detla_caculate_fre(Procedure_matrix(jkk,1),Procedure_matrix(jkk,2),T(4,jkk));
end
    
TH=zeros(1,d2);
tempgogo=0;
for I=ceil(part1*N)+1:N
    tempgogo=tempgogo+1;
    
    pika=pk_test2(Procedure_matrix,d2,k2);
    
    TH2=zeros();
    
    TH3=zeros();     
    
    TH=T(3,:);
    
    TH2=pika.*T(4,:);
    
    TH3=TH2+TH;
    TH4=TH3/sum(TH3);
    H=zeros(1,d2);
  
    for kk=1:d2
        H(kk,1)=sum(TH4(1,1:kk));
    end
    h_temp=rand(1,1);
    No=min(find(H>=h_temp));
   % No=min(h0);
  
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
     if rem(tempgogo,10)==0
    T(3,No)=detla_caculate(Procedure_matrix(No,1),Procedure_matrix(No,2),Pk);
    T(4,jkk)=detla_caculate_fre(Procedure_matrix(jkk,1),Procedure_matrix(jkk,2),T(4,jkk));
     end
end


Time_sum=Procedure_matrix(:,2)'+Procedure_matrix(:,1)';
Time_sum_1=Procedure_matrix(:,2)';
final_fre=(Time_sum_1-Time_sum*q)./((p-q)*Time_sum);
[~,idx]=sort(final_fre);
T_temp=T(:,idx);
result=final_fre;
end