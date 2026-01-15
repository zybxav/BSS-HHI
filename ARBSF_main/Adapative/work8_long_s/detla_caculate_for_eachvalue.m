function result=detla_caculate_for_eachvalue(No_0,No_1)
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