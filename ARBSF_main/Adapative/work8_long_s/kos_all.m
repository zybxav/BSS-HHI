clear all;
format short
opts = optimoptions('fsolve', 'Display', 'off');
filename = 'kosarak_input.txt';
delay=5;




for pp=1:4

    k2_vector = [16,32,48,64,80,96,112,128]
    for tt=1:10
        for jj=1:8
            k2=k2_vector(jj);
             epsilon=2;
             k0=k2*10;
             ground_truth_sta
             N=num_rows-floor(num_rows/2)+1;
             phase2_pre(d2,item,N,k2,delay,Ground_truth,unique_values,filename,epsilon);
        end
    end 
end



 
for pp=1:10

    k2_vector = [32]
    for tt=1:10
        for jj=1:1
            k2=k2_vector(jj);
             epsilon=1;
             k0=k2*10;
             ground_truth_sta
             N=num_rows-floor(num_rows/2)+1;
             phase2_pre(d2,item,N,k2,delay,Ground_truth,unique_values,filename,epsilon);
        end
    end 
    for tt=1:10
        for jj=1:1
            k2=k2_vector(jj);
             epsilon=1.5;
             k0=k2*10;
             ground_truth_sta
             N=num_rows-floor(num_rows/2)+1;
             phase2_pre(d2,item,N,k2,delay,Ground_truth,unique_values,filename,epsilon);
        end
    end
    for tt=1:10
        for jj=1:1
            k2=k2_vector(jj);
             epsilon=2;
             k0=k2*10;
             ground_truth_sta
             N=num_rows-floor(num_rows/2)+1;
             phase2_pre(d2,item,N,k2,delay,Ground_truth,unique_values,filename,epsilon);
        end
    end
    for tt=1:10
        for jj=1:1
            k2=k2_vector(jj);
             epsilon=2.5;
             k0=k2*10;
             ground_truth_sta
             N=num_rows-floor(num_rows/2)+1;
             phase2_pre(d2,item,N,k2,delay,Ground_truth,unique_values,filename,epsilon);
        end
    end
    for tt=1:10
        for jj=1:1
            k2=k2_vector(jj);
             epsilon=3;
             k0=k2*10;
             ground_truth_sta
             N=num_rows-floor(num_rows/2)+1;
             phase2_pre(d2,item,N,k2,delay,Ground_truth,unique_values,filename,epsilon);
        end
    end 
    for tt=1:10
        for jj=1:1
            k2=k2_vector(jj);
             epsilon=3.5;
             k0=k2*10;
             ground_truth_sta
             N=num_rows-floor(num_rows/2)+1;
             phase2_pre(d2,item,N,k2,delay,Ground_truth,unique_values,filename,epsilon);
        end
    end
    for tt=1:10
        for jj=1:1
            k2=k2_vector(jj);
             epsilon=4;
             k0=k2*10;
             ground_truth_sta
             N=num_rows-floor(num_rows/2)+1;
             phase2_pre(d2,item,N,k2,delay,Ground_truth,unique_values,filename,epsilon);
        end
    end
end