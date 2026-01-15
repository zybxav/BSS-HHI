function pika=pk_test2(Procedure_matrix,d2,k2,Pk)
    TT=zeros();
    TT=Procedure_matrix(:,2)'./(Procedure_matrix(:,1)+Procedure_matrix(:,2))';
    [~,idx]=sort(TT);
    pk=TT(idx(d2-k2+1));
    pika=zeros(1,d2);
    for qq=1:d2
        if TT(qq)>=Pk
        pika(qq)=1;
        end
    end
end