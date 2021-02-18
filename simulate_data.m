clc;
clear all;
%%%Generate multiple time series using Euler Maruyama method%%%%%%
rng('shuffle')
global  r k ;
%k1=11;   %k1=100; 
n_max=400;    %%%%Number of time steps
%-------------------------------------------------------------------------------
% varying parameter
c=1; r=1;      
%c_max=1.6;    %%% control parameter considered prior to tipping
c_max=3;
b=1;
da=(c_max-c)/n_max;
%p=[];q=[];
n = 0;
t=0;t_max=300; dt=0.01;
h=1;
M=1;           %% Number of time series to be generated for the given parameter setup%%%%%%%55
corr=0.0;      %%%%%%%Correlation time (change the value of corr to generate time series when system is perturbed with colored noise of different amplitude
     disp(n_max);
     %DETERMINISTIC%
  % while t < t_max 
         %x0=90+rand(1,1);    %% initial condition
    %     a=[r1*x1*(1-x1/(k1))  (c*x1)];
        % x1=x1+(a(1)-a(2)).*dt;
        % fprintf(fileID1, '%f %f %f\n',c, t, x1);
         %t=t+dt; 
   % end
      %STOCHASTIC%
      for j=1:M
          k=5.2;%+(10-8)*rand(1);
          sigma=0.10%+(0.15-0.10)*rand(1);
          x0=70+rand(1,1);
          x=x0;
          fid=fopen(sprintf('model1_%d.dat',j), 'w');
          n=0;
          c=1;
       while n<n_max
             n=n+1;
             disp(n);
             t=0.0;
             c=c+da;
             y=randn(1,1);
             while t < t_max
                    aa=r*x*(1-(x/k));
                    aaa=(c*x^2/(1+x^2));
                    %aaa=(s*x);
                    x=x+(aa-aaa).*dt+sigma.*randn(1,1).*x.*sqrt(dt)*y;  %%%%model equation (replace model equations within while loop to generate time series corresponding to other models
                    jj3=randn(1,1);
                    y=corr*y+(sqrt(1-corr^2))*jj3;
                    t=t+dt;
              end               
               %c1=c(1:200);
               %x1(:,j)=x(1:200,j);
               %fwrite(c,t,x(j))                
              fprintf(fid, '%f\n',x);               
       end
          %c2(j,:)=c(j,1:200);
          %x1(j,:)=x(j,1:200);
      end
       fclose(fid);
% vizualization of time series       
load model1_1.dat
v2=model1_1
plot(1:400,v2(:,1))
%v3=v2(:,3);

