H=size(A);
h1=H(1);
h2=H(2);
AA=zeros(h1,Max_A);
for ik=1:h1
    for ij=1:h2
        temp0=A(ik,ij);
        if temp0==0
        else
            AA(ik,temp0)=1;
        end 
    end
end

item=AA