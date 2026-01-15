function n11=n1i_caculation(Procedure_matrix,N,d,k)
Pmatrix=sum(Procedure_matrix');
e0=zeros();
for i=1:d
  e0(i)=power(log(10)/(2*Pmatrix(i)),1/2);
end
e00=sum(e0);

TT=Procedure_matrix(:,2)'./(Procedure_matrix(:,1)+Procedure_matrix(:,2))';
[Pk,Pk1,~,~]=pk_test(Procedure_matrix,d,k);
Time=zeros();
parfor jkk=1:d
                if TT(jkk)>=Pk1
                    Time(1,jkk)=detla_caculate(Procedure_matrix(jkk,1),Procedure_matrix(jkk,2),Pk);
                else
                    Time(1,jkk)=detla_caculate(Procedure_matrix(jkk,1),Procedure_matrix(jkk,2),Pk1);
                end
end

n11 = sym('n11');
n22 = N - n11;
Pmatrix2 = Pmatrix + n11 * Time;

% 使用数值计算替代符号计算
e1 = sqrt(log(10) ./ (2 * Pmatrix2));
e11 = sum(e1);
Y = n11 * e00 + n22 * e11;
dy = diff(Y, n11);

% 使用数值求解方法
n11 = fsolve(@(n11) double(subs(dy, n11)), 0.5);
% syms n11
% syms n22
% Pmatrix2=Pmatrix+n11*Time;
% n22=N-n11;
% syms e1
% for i=1:d
%   e1(i)=power(log(10)/(2*Pmatrix2(i)),1/2);
% end
% e11=sum(e1);
% Y=n11*e00+n22*e11;
% dy=diff(Y);
% n11=solve(dy);
end

 

