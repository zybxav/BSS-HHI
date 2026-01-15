function phase2_compare(d2,epsilon_index,item,N,k2,delay,Ground_truth,unique_values,filename)

epsilon=epsilon_index;
part=n0_caculate(N,d2,epsilon);
if part>N/d2
    part=N/d2;
end

v11=zeros;
v22=zeros;
v33=zeros;
v44=zeros;
v55=zeros;

t11=zeros;
% t22=zeros;
% t33=zeros;
% t44=zeros;
time_Hu=1;
fileoutput_result='syn131_pred_result_e2.txt';
fileoutput_pred_distribution='syn131_pred_topk_items_e2.txt';
fileoutput_gt_distribution='syn131_gt_topk_items_e2.txt';
parfor Hu=1:1:time_Hu
    
    [v1,v2,v3,v4,v5,t1,bandit_topk, gt_topk]=phase2_process(d2,epsilon_index,item,N,k2,Ground_truth,part,delay,Hu,unique_values);
    fileID = fopen(fileoutput_result,'a');   % 打开并追加写入
    v1
    v2
    v3
    v4
    v5
    fprintf(fileID, ...
    '[eps=%d,k=%d,iterations=%d] ncr=%.4f,mse=%.4f,acc=%.4f,f1=%.4f,mcr=%.4f\n', ...
    epsilon_index, k2, Hu, v1,v2,v3,v4,v5);
    fclose(fileID);
    fileID = fopen(fileoutput_pred_distribution,'a');
    fprintf(fileID, '[eps=%d] ', epsilon_index);
    fprintf(fileID, '[k=%d] ', k2);
    fprintf(fileID, '[iter=%d] ', Hu);
    fprintf(fileID, '(%d,%d) ', bandit_topk');
    fprintf(fileID, '\n');
    fclose(fileID);

    % ---- 保存真实值 ----
    fileID = fopen(fileoutput_gt_distribution,'a');
    fprintf(fileID, '[eps=%d] ', epsilon_index);
    fprintf(fileID, '[k=%d] ', k2);
    fprintf(fileID, '[iter=%d] ', Hu);
    fprintf(fileID, '(%d,%d) ', gt_topk');
    fprintf(fileID, '\n');
    fclose(fileID);

    v11=v11+v1;
    v22=v22+v2;
    v33=v33+v3;
    v44=v44+v4;
    v55=v55+v5;

    t11=t11+t1;
    % t22=t22+t2;
    % t33=t33+t3;
    % t44=t44+t4;
end
 %v11、v22、v33代表三种不同的指标
v11=v11/time_Hu;
v22=v22/time_Hu;
v33=v33/time_Hu;
v44=v44/time_Hu;
v55=v55/time_Hu;
%四种算法运行时间，此处删去t22,t33,t44，因为我只计算了bandit算法，没有其他三种
t11=t11/time_Hu;
% t22=t22/time_Hu;
% t33=t33/time_Hu;
% t44=t44/time_Hu;
% v=[v11,v22,v33,v44];


end

