%% k:carrying capacity;r:maximum growth rate; c:maximum grazing rate; b:half saturation constant 
%% Generate multiple time series using Euler Maruyama method%%%%%%
%% M:Number of time series to be generated for the given parameter
%% n_max:number of timesteps;x0:initial condition. sigma and k are varied within a range in order to add variability across time series
%% corr:Correlation time (change the value of corr to generate time series when system is perturbed with colored noise of different amplitude
%% generate time series with increased variablility



function x=model1(n_max,c,c_max,r,b,M,corr)
% Generate multiple stochastic time series by solving the following differential equation using the Euler Maruyama method:
%
% :math:`\frac{dN}{dt} = rN(1-\frac{N}{K})-\frac{cN^{2}}{b^{2}+N^{2}}`. 
% 
% The values of `sigma` and `k` are varied within a range in order to add variability across time series to generate time series with increased variablility. 
% 
% Change the value of corr to generate time series when system is perturbed with colored noise of different amplitude.
% 
% :param k: carrying capacity
% :param r: maximum growth rate
% :param c: maximum grazing rate
% :param b: half saturation constant 
% :param M: Number of time series to be generated for the given set of parameter values
% :param n_max: number of timesteps
% :param corr: Correlation time 
% :param x0: initial condition. 
%
%
% .. note:: Modify the stochastic differential equation to generate time series data from other models
%


global c r n_max c_max b M corr
rng('shuffle')
%n_max=1000  %%%considered to generate entire timeseries containing the transition point
da=(c_max-c)/n_max;
t=0;t_max=300; dt=0.01;

  %--------------------------------------
  %DETERMINISTIC TIMESERIES GENERATION
  %--------------------------------------
    % while t < t_max 
    %x0=90+rand(1,1);    %% initial condition
    %a=[r1*x1*(1-x1/(k1))  (c*x1)];
    % x1=x1+(a(1)-a(2)).*dt;
    % fprintf(fileID1, '%f %f %f\n',c, t, x1);
    %t=t+dt;
    % end
    
  %-------------------------------------------
  % STOCHASTIC TIMESERIES GENERATION
  %-------------------------------------------
      for j=1:M
          k=5.2+(10-8)*rand(1);
          sigma=0.10+(0.15-0.10)*rand(1);    
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
                    term1  = r*x*(1-(x/k));
                    term2  =(c*x^2/(1+x^2));
                    %term2 =(s*x);

                    %-------------------------------------------------------------------------------------------------- 
                    %                      Model Equation 
                    % =================================================================================================
                    % (replace model equations within while loop to generate time series corresponding to other models)
                    %--------------------------------------------------------------------------------------------------
                    x=x+(term1-term2).*dt+sigma.*randn(1,1).*x.*sqrt(dt)*y;  
                    jj3=randn(1,1);

                    %------------------------------------------------------------------------------
                    %  (incorporation of stochasticity corresponding to different colored noise)
                    %------------------------------------------------------------------------------
                    y=corr*y+(sqrt(1-corr^2))*jj3;    
        
                    t=t+dt;
        
              end
              fprintf(fid, '%f\n',x);     
       end
      end
      fclose(fid);