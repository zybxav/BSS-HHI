function result=detla_caculate_topk(No_0,No_1,Pk)
     syms delta0
     t_time=No_0+No_1;
     tempo=zeros(1,t_time);
     if No_1>=1
     tempo(1:No_1)=1;
      variance0=var(tempo);
     else
      variance0=0;
     end
    
     %求当前的均值
     Mean0=(No_1)/t_time;
    %这里应该有个最小的统计单位，单独一个是没办法统计，这里要设定一个最小的单位
    %每收集一次就更新一个有点太过分。
    theta0=abs(Mean0-Pk);
   % result=solve((power(variance0*2*log(1.5/delta0)/t_time,1/2))+3*1*log(1.5/delta0)/t_time-theta0==0,delta0);
    result=solve((power(variance0*2*log(3/delta0)/t_time,1/2))+3*1*log(3/delta0)/t_time-theta0==0,delta0);
    %这个是错误率，按照错误率越大，sample到的概率越大
end