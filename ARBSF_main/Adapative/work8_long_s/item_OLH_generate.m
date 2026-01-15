function item_grr=item_OLH_generate(N,item,ti,d)
for i=1:N
    itemi=item(i,:);
   % ti=sum(sum(itemi~=0));
          H=find(itemi~=0);
         a=randperm(length(H));
         a1=a(1);
     item_grr(i)=itemi(a1);
end
end