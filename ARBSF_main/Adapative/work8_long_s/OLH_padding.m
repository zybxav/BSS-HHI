function result=OLH_padding(d,e,item,N,ti)
d2=d+1;
g=round(e)+1;
%然后就是扰动g。对应的hash函数是有个表把。。。什么鬼玩意儿
p_olh=(e)/(e+g-1);
q_olh=(1)/(e+g-1);
%randomly pick up a hash function
D=1+round((g-1)*rand(1,1));
%恢复算法就是用p*and q* is enough
p_olh_star=(e)/(e+g-1);
q_olh_star=1/g;

%GRR版本,这个要修改，这个好像是等价的
po_olh=e/(e+g-1);
qo_olh=1/(e+g-1);
pertured_vector_OLH=zeros(1,d+1);
temp_prime=round(1.5*power(N,1/2));
max_prime=max(primes(temp_prime));
Ta=randperm(max_prime-1);
Tb=randperm(max_prime);
TTab=zeros();
aq=0;

for ta=1:(max_prime-1)
    for tb=1:max_prime
        aq=aq+1;
   TTab(aq,1)=Ta(ta);
   TTab(aq,2)=Tb(tb);
    end
end
temp_prime=round(1.5*power(N,1/2));
max_prime=max(primes(temp_prime));
Ta=randperm(max_prime-1);
Tb=randperm(max_prime);
TTab=zeros();
aq=0;
for ta=1:(max_prime-1)
    for tb=1:max_prime
        aq=aq+1;
   TTab(aq,1)=Ta(ta);
   TTab(aq,2)=Tb(tb);
    end
end

Tb=randperm((max_prime-1)*max_prime);
for i=1:N    
    i_rand=Tb(i);
    a=TTab(i_rand,1);
    b=TTab(i_rand,2);
  % pick up one item
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      itemi=item(i,:);
    H=find(itemi==1);
    %%%%%%
      if length(H)>=ti
          random_num = H(randperm(numel(H),ti));
          else
          random_num=zeros(1,ti)+d2;
          random_num(1:length(H))=H;
      end
     temp_num = randperm(ti,1);
     r=random_num(temp_num);
        
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % generate a hash function    
     %use the user numer as the seed
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
     %扰动回去的位置
     %pertured_vector_OLH(temp3)=pertured_vector_OLH(temp3)+1;
     for jj1=1:d+1
         %将每个位置代入hash 函数，求结果
         result_temp=rem(rem(a*jj1+b,max_prime),g)+1;
         if result_temp==temp3
         pertured_vector_OLH(jj1)=pertured_vector_OLH(jj1)+1;
         end
     end
end
% Actually, each user update only one data
%result=(pertured_vector_OLH-(N/ti)*q_olh_star)/((p_olh_star-q_olh_star)*(N/ti));
%result=(pertured_vector_OLH-(N)*q_olh_star)/((p_olh_star-q_olh_star)*(N));
A=g/(p_olh*g-1);
B=N/(p_olh*g-1);
result=(pertured_vector_OLH*A-B)/N;
result=result(1:d);
end