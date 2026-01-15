function result=variance_caculate_fre(No_0,No_1)
     syms delta0
     t_time=No_0+No_1;
     tempo=zeros(1,t_time);
     if No_1>=1
     tempo(1:No_1)=1;
      variance0=var(tempo);
     else
      variance0=0;
     end
end