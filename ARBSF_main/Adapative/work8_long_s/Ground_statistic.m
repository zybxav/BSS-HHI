function Ground_truth=Ground_statistic(A,h1,h2,d)
Ground_truth=zeros(1,d);
for kk=1:h1
    for ii=1:h2
        temp=A(kk,ii);
        if temp~=0
           Ground_truth(temp)=Ground_truth(temp)+1;
        end
    end
end
end