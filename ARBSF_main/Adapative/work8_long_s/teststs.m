% V_B=zeros(10,24);
% epsilon=0.25;
% e=exp(epsilon);
% phase2_compare
% V_B(1,:)=v;


p=e/(e+1);
q=1/(e+1);
T=zeros(4,d2);
for o=1:d2;
    T(1,o)=o;
    T(3,o)=1/d2;  %这个是均分的概率
end
tempgogo=0;
Procedure_matrix=zeros(d2,2);
for I=1:ceil(0.1*N)
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

%这里我们先暂时设定，误差是扰动后的数据带来的？而不是扰动前的数据
Pk=pk_test(Procedure_matrix,d2,k2);
    
for jkk=1:d2
    T(3,jkk)=detla_caculate(Procedure_matrix(jkk,1),Procedure_matrix(jkk,2),Pk);
end
 

TH=zeros(1,d2);
for I=ceil(0.1*N)+1:N
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
      if tempgogo==floor(60000*(0.1+0.25))
          Pk=pk_test(Procedure_matrix,d2,k2);
    T(3,No)=detla_caculate(Procedure_matrix(No,1),Procedure_matrix(No,2),Pk);
      end
end
  Pk=pk_test(Procedure_matrix,d2,k2);
  delta0=vpa(1-prod(1-Pk))
