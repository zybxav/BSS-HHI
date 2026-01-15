function [pk,pk_1,pika,result]=pk_test(Procedure_matrix,d2,k2)
    TT=zeros();
    TT=Procedure_matrix(:,2)'./(Procedure_matrix(:,1)+Procedure_matrix(:,2))';
    [~,idx]=sort(TT);
    pk=TT(idx(d2-k2+1));
    pk_1=TT(idx(d2-k2+1+1));
    pika=zeros(1,d2);
    for qq=1:d2
        if TT(qq)>=pk
        pika(qq)=1;
        end
    end
    %再返回一个theta，加在这里
    %对应的方差是
     No_0=Procedure_matrix(idx(d2-k2+1),1);
     No_1=Procedure_matrix(idx(d2-k2+1),2);
     t_time=No_0+No_1;
     tempo=zeros(1,t_time);
     variance0=0;
     if No_1>=1
     tempo(1:No_1)=1;
      variance0=var(tempo);
     else
      variance0=0;
     end
     %求当前的均值
     Mean0=(No_1)/t_time;
     delta0=1;
    result=(power(variance0*2*log(3/delta0)/t_time,1/2))+3*1*log(3/delta0)/t_time;

end