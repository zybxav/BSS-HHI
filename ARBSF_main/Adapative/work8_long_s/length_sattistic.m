function ti=length_sattistic(N,item,d)
dbstop if error
Np1=floor(0.1*N);

L=zeros(1,d);
L_item_original=zeros(1,Np1);
L_item_perturb=zeros(1,Np1);
for i=1:Np1
    itemi=item(i,:);
    h1=length(find(itemi==1));
    if h1~=0
    L_item(1,i)=h1;
    L(h1)=L(h1)+1;
    end
end
%perturbed version
epsilon=10;
e=exp(epsilon);
av=size(L_item);
Np1=av(1);

L_item_perturb_GRR=GRR_single(d,e,L_item,Np1);
L_item_perturb_OLH=OLH_single(d,e,L_item,Np1);

%what is 90th percent
%sum value >=0.9
L_item_sum_GRR=zeros(1,d);
L_item_sum_OLH=zeros(1,d);
ij=1;
 L_item_sum_GRR(1)=L_item_perturb_GRR(1);
for ij=2:d
    L_item_sum_GRR(ij)=L_item_sum_GRR(ij-1)+L_item_perturb_GRR(ij);
    L_item_sum_OLH(ij)=L_item_sum_OLH(ij-1)+L_item_perturb_GRR(ij);
end
M_Max_GRR=max(find(L_item_sum_GRR<0.9))+1;
M_Max_OLH=max(find(L_item_sum_OLH<0.9))+1;
ti=M_Max_OLH;
end

