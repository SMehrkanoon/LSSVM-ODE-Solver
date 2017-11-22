%************************************************************************
% LSSVM_ODE_Demo: 
%
% Learning solution of ODEs using Least Squares Support Vector Machines
%
% Created by
%     Siamak Mehrkanoon
%     Dept. of Electrical Engineering (ESAT)
%     Research Group: STADIUS
%     KU LEUVEN
%
% (c) 2012
%************************************************************************

% Citations:

%[1] Mehrkanoon S., Falck T., Suykens J.A.K., 
%"Approximate Solutions to Ordinary Differential Equations Using Least Squares Support Vector Machines",
%IEEE Transactions on Neural Networks and Learning Systems, vol. 23, no. 9, Sep. 2012, pp. 1356-1367.


%[2] Mehrkanoon S., Suykens J.A.K.,
%"LS-SVM approximate solution to linear time varying descriptor systems", 
%Automatica, vol. 48, no. 10, Oct. 2012, pp. 2502-2511.


%[3] Mehrkanoon S., Suykens J.A.K., 
%"Learning Solutions to Partial Differential Equations using LS-SVM",
%Neurocomputing, vol. 159, Mar. 2015, pp. 105-116.


%Author: Siamak Mehrkanoon

%%  ================= Example (1)  ===================

%   dy        1+ 3 t^2                            1+ 3 t^2
%  ----  + (t+ ---------- ) y(t) = t^3 + 2t + t^2 --------------
%   dt         1+ t + t^3                         1+ t + t^3
%
% 0 <=  t  < = 1
%
% Initial Condition
% y(0) = 1

%  ============ Example (2)  ===================

%   dy     
%  ----  + 2 y(t) = t^3 * sin(t/2) 
%   dt     
%
% 0 <=  t  < = 10
%
% Initial Condition
% y(0) = 1




%% ====================================================

clear all; close all; clc
warning('off','all')


%========= Example (1) ================
t0=0;  
tf=1;
np= 2^4;  %  Number of training points 
t= linspace(t0,tf,np);
s=size(t,2);
h= (tf-t0)/np;
g=@(x)  x.^3+ 2*x + x.^2 .* ((1+3*x.^2) ./ (1+ x+ x.^3 )) ;
q=@(x) - (x+ (1+3*x.^2)./(1+x+x.^3));
qq=q(t');
qq=qq(2:end);
a= 1; % initial condition
func=@(r) exp((-r.^2)/2)./(1+ r + r.^3) +r.^2; % Exact solution

% ============ Example (2) ============
% t0=0;  
% tf=10;
% np= 2^4;  %  Number of training points 
% t= linspace(t0,tf,np);
% s=size(t,2);
% h= (tf-t0)/np;
% g= @(x) x.^3 .* sin(x./2);
% q= -2*ones(s,1);
% qq=q(2:end);
% a=1;  % initial condition
% exact = 'Dy=-2*y+tt^3*sin(tt/2)'; % Exact solution
% y = dsolve(exact,'y(0)=1','tt');


% =================================

t=t';
pt_training=t(2:end); % training points
pt_validation= pt_training-(h/2); % validation points
pt_test=t0:0.04:tf; pt_test=pt_test';  % Test points



%=======  Example (1) ===============
exact_sol_training= func(pt_training);  % for Example (2)
exact_sol_test= func(pt_test); % for Example (2)
qval = q(pt_validation);


%=======  Example (2) ===============
% tt=pt_training;  exact_sol_training=eval(vectorize(y));  % for Example (1)
% tt=pt_test; exact_sol_test=eval(vectorize(y)); % for Example (1)
% qval= -2*ones(size(pt_validation,1),1);

%% Tuning the model parameters (sigma and gamma)

ng_sig=10;  ng_gam=10;
sigma_range = logspace(-3,3,ng_sig);
gamma_range=logspace(0,15,ng_gam);
BB=zeros(ng_gam,ng_sig);
for gamma_idx =1:size(gamma_range,2)
    gamma = gamma_range(gamma_idx);   
    for sig_idx=1:size(sigma_range,2)
        sig = sigma_range(sig_idx);
        K=KernelMatrix(pt_training,'RBF_kernel',sig);
        xx1=pt_training*ones(1,size(pt_training,1));
        xx2=pt_training*ones(1,size(pt_training,1));
        cof1=-2*(xx1-xx2')/(sig);
        Kp1=(cof1.*K)';
        Kp1=diag(qq)*Kp1;
        xx3=pt_training*ones(1,size(pt_training,1));
        xx4=pt_training*ones(1,size(pt_training,1));
        coff1=2*(xx3-xx4')/(sig);
        Kp2=(coff1.*K)';
        Kp2=Kp2* diag(qq);
        D1=diag(qq);
        KK=D1*K*D1';
        Kpp=(2/sig)*K - (cof1.^2) .* K;
        KC= Kpp -Kp1-Kp2 +  KK;
        m1=size(KC,1);
        K2=KernelMatrix(pt_training,'RBF_kernel',sig,t(1));
        xx1=pt_training*ones(1,size(t(1),1));
        xx2=t(1)*ones(1,size(pt_training,1));
        cof2=-2*(xx1-xx2')/(sig);
        K3=cof2.*K2;
        K22=qq.*K2;
        K4=K3-K22;
        A=[ KC+1/gamma*eye(m1) , K4, -1*qq;...
            K4' , 1, 1;...
            -1*qq', 1, 0];
        B=[g(pt_training); a ; 0 ];
        result=A\B;
        alpha=result(1:m1);
        beta=result(end-1);
        b=result(end);
        Kt=KernelMatrix(pt_training,'RBF_kernel',sig,pt_validation);
        xx1=pt_training*ones(1,size(pt_validation,1));
        xx2=pt_validation*ones(1,size(pt_training,1));
        cof3=-2*(xx1-xx2')/(sig);
        Kxt= cof3 .* Kt;
        K2t=KernelMatrix(pt_validation,'RBF_kernel',sig,t(1));
        yUval=(Kxt' - Kt'*diag(qq) )* alpha   +  K2t* beta  + b;
        xx3=pt_validation*ones(1,size(t(1),1));
        xx4=t(1)*ones(1,size(pt_validation,1));
        cof4=2*(xx3-xx4')/(sig);
        K1yt = - cof4 .* K2t;
        Kxyt=(  ( (2/sig) - (cof3.^2) ) .* Kt);
        Kyt= - cof3 .* Kt ;
        yprimeUval=(Kxyt'  - Kyt'*diag(qq) )* alpha   +  K1yt* beta ;
        BB(gamma_idx,sig_idx)=mse(yprimeUval - diag(qval)*  yUval - g(pt_validation)); 
    end   
end
[temp,ind_col]=min(BB,[],2); [temp2,ind_row]=min(temp);
p=ind_row ; q=ind_col(ind_row) ;
gamma=gamma_range(p); sig=sigma_range(q);

% Calculating the training computational time
count=1; elapsed_time=cputime;
while count<100  
    K=KernelMatrix(pt_training,'RBF_kernel',sig);
    xx1=pt_training*ones(1,size(pt_training,1));
    xx2=pt_training*ones(1,size(pt_training,1));
    cof1=-2*(xx1-xx2')/(sig);
    Kp1=(cof1.*K)';
    Kp1=diag(qq)*Kp1;
    xx3=pt_training*ones(1,size(pt_training,1));
    xx4=pt_training*ones(1,size(pt_training,1));
    coff1=2*(xx3-xx4')/(sig);
    Kp2=(coff1.*K)';
    Kp2=Kp2* diag(qq);
    D1=diag(qq);
    KK=D1*K*D1';
    Kpp=(2/sig)*K - (cof1.^2) .* K;
    KC= Kpp -Kp1-Kp2 +  KK;
    m1=size(KC,1);
    K2=KernelMatrix(pt_training,'RBF_kernel',sig,t(1));
    xx1=pt_training*ones(1,size(t(1),1));
    xx2=t(1)*ones(1,size(pt_training,1));
    cof2=-2*(xx1-xx2')/(sig);
    K3=cof2.*K2;
    K22=qq.*K2;
    K4=K3-K22;
    A=[ KC+1/gamma*eye(m1) , K4, -1*qq;...
        K4' , 1, 1;...
        -1*qq', 1, 0];
    B=[g(pt_training); a ; 0 ];
    result=A\B;
    alpha=result(1:m1);
    beta=result(end-1);
    b=result(end);
    Kp=cof1.*K;
    yhat_train=(Kp' - K*diag(qq)   )* alpha   +  K2* beta  + b;
    count=count+1;
end
total_time= cputime - elapsed_time;
training_time= total_time /100;
Absolute_error_training=norm(yhat_train-exact_sol_training,inf);
MSE_training=mse(yhat_train-exact_sol_training);
fprintf('-------  training set ------------------\n\n')
fprintf('Training computational time=%f\n',training_time)
fprintf('Max Abs Error on training set=%d\n',Absolute_error_training)
fprintf('MSE on training set=%d\n\n',MSE_training)


%% =============================================
Kt=KernelMatrix(pt_training,'RBF_kernel',sig,pt_test);
xx1=pt_training*ones(1,size(pt_test,1));
xx2=pt_test*ones(1,size(pt_training,1));
cof1=-2*(xx1-xx2')/(sig);
Kpt= cof1 .* Kt;
K2t=KernelMatrix(pt_test,'RBF_kernel',sig,t(1));
yhat_test=(Kpt' - Kt'*diag(qq) )* alpha   +  K2t* beta  + b;
Absolute_error_test=norm(yhat_test-exact_sol_test,inf);
MSE_test=mse(yhat_test-exact_sol_test);
fprintf('-------  test set ----------------------\n\n')
fprintf('Max Abs Error on test set=%d\n',Absolute_error_test)
fprintf('MSE on test set=%d\n\n',MSE_test)
fprintf('-------  Finished -----------------------\n\n')


%%  Final Plotting

figure
plot(pt_test,exact_sol_test,'ro','markersize',6)
hold on
plot(pt_test,yhat_test,'b')
hleg1=legend('Exact solution','Learned solution using LSSVM');
set(hleg1,'Location','SouthWest');
title(['MSE on test set=',num2str(MSE_test)],'FontSize',20)
set(gca,'FontSize',20)
xlabel('t')
ylabel('y(t)')
grid on