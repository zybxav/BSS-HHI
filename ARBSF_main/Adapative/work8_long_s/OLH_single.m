function result=OLH_single(d,e,item,N)
g=round(e)+1;
po_olh=e/(e+g-1);
qo_olh=1/(e+g-1);
pertured_vector_OLH=zeros(1,d);

temp_prime=round(1.5*power(N,1/2));
max_prime=max(primes(temp_prime));
Ta=randperm(max_prime-1);
Tbb=randperm(max_prime);
TTab=zeros();
aq=0;
for ta=1:(max_prime-1)
    for tb=1:max_prime
        aq=aq+1;
   TTab(aq,1)=Ta(ta);
   TTab(aq,2)=Tbb(tb);
    end
end
Tb=randperm((max_prime-1)*max_prime);
for i=1:N    
    i_rand=Tb(i);
    a=TTab(i_rand,1);
    b=TTab(i_rand,2);
    r=item(i);
     resulti=rem(rem(a*r+b,max_prime),g)+1;
    probobilty_vector3=zeros(1,g);
    r2=1;
    if resulti==1
        probobilty_vector3(1)=po_olh;
    else
         probobilty_vector3(1)=qo_olh;
    end
    for r2=2:g
    if r2==resulti
       probobilty_vector3(r2)=probobilty_vector3(r2-1)+po_olh;
    else    
       probobilty_vector3(r2)=probobilty_vector3(r2-1)+qo_olh;
    end
    end
     temp1=rand(1,1);
     temp3=min(find(probobilty_vector3>=temp1));
     for jj1=1:d
         result_temp=rem(rem(a*jj1+b,max_prime),g)+1;
         if result_temp==temp3
         pertured_vector_OLH(jj1)=pertured_vector_OLH(jj1)+1;
         end
     end
end
A=g/(po_olh*g-1);
B=(N)/(po_olh*g-1);
result=(pertured_vector_OLH*A-B)/N;
end