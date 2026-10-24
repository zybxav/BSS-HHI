function result = detla_caculate(No_0, No_1, Pk2, T)
    t_time = No_0 + No_1;
    tempo = ones(1, No_1);
    variance0 = var(tempo, 1);
    Mean0 = No_1 / t_time;
    theta00 = abs(Mean0 - Pk2);
    
    fun = @(x) (sqrt(variance0 * 2 * log(3 / x) / t_time) + 3 * 1 * log(3 / x) / t_time - theta00);
    delta0 = fminbnd(fun, 1e-6, 1);
    
    if isempty(delta0) || ~isfinite(delta0)
        result = T;
    else
        result = delta0;
    end
end
% function result=detla_caculate(No_0,No_1,Pk2,T)
%       syms delta0 result variance0 theta00 delta0
%      t_time=No_0+No_1;
%      tempo=zeros(1,t_time);
%      variance0=0;
%      if No_1>=1
%      tempo(1:No_1)=1;
%       variance0=var(tempo);
%      else
%       variance0=0;
%      end
%      %求当前的均值
%      Mean0=(No_1)/t_time;
%     %这里应该有个最小的统计单位，单独一个是没办法统计，这里要设定一个最小的单位
%     %每收集一次就更新一个有点太过分。
%     theta00=abs(Mean0-Pk2);
%   %  theta00= abs(Mean0-0.2798);
%    % result=solve((power(variance0*2*log(1.5/delta0)/t_time,1/2))+3*1*log(1.5/delta0)/t_time-theta00==0,delta0);
%     result=solve((power(variance0*2*log(3/delta0)/t_time,1/2))+3*1*log(3/delta0)/t_time-theta00==0,delta0);
%     %这个是错误率，按照错误率越大，sample到的概率越大
%     if isempty(result)==1
%       result=solve((power(variance0*2*log(3/delta0)/t_time,1/2))+3*1*log(3/delta0)/t_time-num2str(theta00)==0,delta0);
%     end
%       
%     if isempty(result)==1
%       result=solve((power(num2str(variance0)*2*log(3/delta0)/t_time,1/2))+3*1*log(3/delta0)/t_time-theta00==0,delta0);
%     end    
%     
%     if isempty(result)==1
%         result=T;
%     end
% end