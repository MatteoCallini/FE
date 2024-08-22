% Final Project AA2023-2024
% Project 1b
%
% Model Risk on Regulatory Capital
%
% Callini Matteo
% Delera Giacomo

clear
clc
format long

%% Fix the seeds
% seed_m = randi([1,1e+7]); % seed for the market (we change it every simulation)
seed_m =  150; % seed for the market (fix only to compare with the report)
seed = 10; % seed for Montecarlo simulation

%% Read data from Excel
data = readData("CreditModelRisk_RawData.xlsx");

%% Point 1
% Part a
% Shapiro-Wilk test univariate
fprintf("<strong>-------------------- Point 1 --------------------</strong>\n\n")

% Confidence level
alpha = 1 - 0.999;
LGD = 1 - data.RR;

% We consider the inverse of the standard Gaussian in order to verify 
% gaussianity 
k_SG = norminv(data.DR_SG); % Speculative grade corporate
k_AR = norminv(data.DR_All_rated); % All grade corporate

% Shapiro tests
[H_SG, pValue_SG, W_SG] = swtest(k_SG,alpha);
[H_AR, pValue_AR, W_AR] = swtest(k_AR,alpha);
[H_LGD, pValue_LGD, W_LGD] = swtest(LGD,alpha);

% Shapiro-Wilk test bivariate
[H_AR_LGD, pValue_AR_LGD] = Roystest([k_AR,LGD],alpha);
[H_SG_LGD, pValue_SG_LGD] = Roystest([k_SG,LGD],alpha);

fprintf("Statistical analysis (alpha = %.3f)\n\n",alpha)
fprintf("Shapiro-Wilk test - Univariate case\n")
fprintf("K_SG p-value: %.4f\n",pValue_SG);
fprintf("K_AR p-value: %.4f\n",pValue_AR);
fprintf("Loss given default p-value: %.4f\n\n",pValue_LGD);
fprintf("Royston test - Bivariate case\n")
fprintf("K_AR - Loss given default p-value: %.4f\n",pValue_AR_LGD);
fprintf("K_SG - Loss given default p-value: %.4f\n\n",pValue_SG_LGD);

%% Part b
% Pearson coefficients
corr_SG = corrcoef(k_SG,LGD);
corr_AR = corrcoef(k_AR,LGD);

% Pearson coefficients confidence level
alpha = 1-0.95;

% z transformation values
z_SG = log((1+corr_SG(1,2))/(1-corr_SG(1,2)))/2;
z_AR = log((1+corr_AR(1,2))/(1-corr_AR(1,2)))/2;

% size interval
n = length(data.DR_SG);

% z transformation intervals
z_SG_interval = [z_SG - norminv(1-alpha/2)*sqrt(1/(n-3)), z_SG + norminv(1-alpha/2)*sqrt(1/(n-3))];
z_AR_interval = [z_AR - norminv(1-alpha/2)*sqrt(1/(n-3)), z_AR + norminv(1-alpha/2)*sqrt(1/(n-3))];

% Pearson coefficient intervals
corr_SG_interval = [(exp(2*z_SG_interval(1))-1)/(exp(2*z_SG_interval(1))+1),(exp(2*z_SG_interval(2))-1)/(exp(2*z_SG_interval(2))+1)];
corr_AR_interval = [(exp(2*z_AR_interval(1))-1)/(exp(2*z_AR_interval(1))+1),(exp(2*z_AR_interval(2))-1)/(exp(2*z_AR_interval(2))+1)];

fprintf("Pearson coefficient kAR - LGD: %.4f - CI = [%.4f, %.4f]\n",corr_AR(1,2),corr_AR_interval(1),corr_AR_interval(2))
fprintf("Pearson coefficient kSG - LGD: %.4f - CI = [%.4f, %.4f]",corr_SG(1,2),corr_SG_interval(1),corr_SG_interval(2))

%% Plots

% Correlation between k and LGD

% Linear regression coefficients
a_AR = polyfit(k_AR,LGD,1);
a_SG = polyfit(k_SG,LGD,1);

% Linear regression values
k_AR_reg = polyval(a_AR,k_AR);
k_SG_reg = polyval(a_SG,k_SG);

figure('Name','Scatter plot K_AR - LGD with linear regression')
plot(k_AR,LGD,'ro')
title('Scatter plot K_A_R - LGD with linear regression')
hold on
plot(k_AR,k_AR_reg,'blue')

figure('Name','Scatter plot K_SG - LGD with linear regression')
plot(k_SG,LGD,'ro')
title('Scatter plot K_S_G - LGD with linear regression')
hold on
plot(k_SG,k_SG_reg,'blue')

%% Point 2
fprintf("\n\n<strong>--------------------- Point 2 --------------------</strong>\n\n")

% Point a

% compute mean values for naive approach
LGD_hat = mean(1-data.RR); % Mean Loss Given Default
PD_AR_hat = mean(data.DR_All_rated); % Mean All Rated Default probability
PD_SG_hat = mean(data.DR_SG); % Mean Speculative Grade Default probability

% Confidence levels
alpha = [0.999, 0.99];

%% Naive approach

EL = struct();
RC = struct();

% Expected Loss
EL.naive.SG = LGD_hat*PD_SG_hat; % Speculative grade
EL.naive.AR = LGD_hat*PD_AR_hat; % All rated grade

% Regulatory capital in naive approach
% Speculative grade
RC.naive.SG = LGD_hat*normcdf((norminv(PD_SG_hat)-sqrt(correlation_IRB(PD_SG_hat))*...
              norminv(1-alpha))/(sqrt(1-correlation_IRB(PD_SG_hat))))-EL.naive.SG;

% All rated grade
RC.naive.AR = LGD_hat*normcdf((norminv(PD_AR_hat)-sqrt(correlation_IRB(PD_AR_hat))*...
              norminv(1-alpha))/(sqrt(1-correlation_IRB(PD_AR_hat))))-EL.naive.AR;

%% General approach
% Number of simulations
N = 1e+7;

% Speculative k standard deviation
sigma_SG = std(k_SG);

% All rated k standard deviation
sigma_AR = std(k_AR);

% Recovery rate standard deviation
sigma_LGD = std(1-data.RR);

% k hat (k mean obtained by inverting PD) computation
f_SG = @(k) integral(@(x) exp(-x.^2/2)./sqrt(2*pi).*normcdf(k + sigma_SG.*x),-inf,+inf) - PD_SG_hat;
k_hat_SG = fzero(f_SG,mean(k_SG));
f_AR = @(k) integral(@(x) exp(-x.^2/2)./sqrt(2*pi).*normcdf(k + sigma_AR.*x),-inf,+inf) - PD_AR_hat;
k_hat_AR = fzero(f_AR,mean(k_AR));

%% QQ-plots and histograms

figure('Name','Gaussianity k_AR')
subplot(1,2,1)
qqplot(k_AR)
title('QQ-plot k_A_R')
subplot(1,2,2)
histogram(k_AR, 'Normalization', 'pdf')
title('Density - Histogram K_A_R')
hold on
plot(linspace(min(k_AR),max(k_AR),100),(1/(sigma_AR*sqrt(2*pi)))*exp(-0.5*((linspace(min(k_AR)-0.1,max(k_AR)+0.1,100)-k_hat_AR)/sigma_AR).^2))

figure('Name','Gaussianity k_SG')
subplot(1,2,1)
qqplot(k_SG)
title('QQ-plot k_S_G')
subplot(1,2,2)
histogram(k_SG,7,'Normalization', 'pdf')
title('Density - Histogram K_S_G')
hold on
plot(linspace(min(k_SG),max(k_SG),100),(1/(sigma_SG*sqrt(2*pi)))*exp(-0.5*((linspace(min(k_SG)-0.1,max(k_SG)+0.1,100)-k_hat_SG)/sigma_SG).^2))

figure('Name','Gaussianity LGD')
subplot(1,2,1)
qqplot(LGD)
title('QQ-plot LGD')
subplot(1,2,2)
histogram(LGD,7,'Normalization', 'pdf')
title('Density - Histogram LGD')
hold on
plot(linspace(min(LGD),max(LGD),100),(1/(sigma_LGD*sqrt(2*pi)))*exp(-0.5*((linspace(min(LGD)-0.1,max(LGD)+0.1,100)-LGD_hat)/sigma_LGD).^2))

%% Market parameters simulation
rng(seed_m)
M = randn(1,N);

%% k and LGD terms simulation
rng(seed)
g = randn(2,N); % LGD and K standard gaussian simulation

%% Add-on
AddOn = struct();
AddOn_CI = struct();
EL_CI = struct();
RC_CI = struct();
% Montecarlo simulation
% We consider both alpha in order to compare them with the same parameters
% fprintf("\n<strong>------------- Montecarlo Regulatory capital LHP Stress - Add on computation ----------------<strong>\n\n")

% LGD simulation, fixing K     
tic
[~, EL.Normal.SG.LGD, RC.Normal.SG.LGD, RC_CI.Normal.SG.LGD, EL_CI.Normal.SG.LGD] = montecarlo_RC(N,norminv(PD_SG_hat),sigma_SG,LGD_hat,sigma_LGD,alpha,g,M,1,corr_SG(1,2));
[~, EL.Normal.AR.LGD, RC.Normal.AR.LGD, RC_CI.Normal.AR.LGD, EL_CI.Normal.AR.LGD] = montecarlo_RC(N,norminv(PD_AR_hat),sigma_AR,LGD_hat,sigma_LGD,alpha,g,M,1,corr_AR(1,2));
fprintf("LGD simulation - Montecarlo simulations computation time: ")
toc
% Add-on computation
[AddOn.Normal.SG.LGD, AddOn_CI.Normal.SG.LGD] = Add_on(RC.Normal.SG.LGD,RC.naive.SG,EL.Normal.SG.LGD,EL.naive.SG,RC_CI.Normal.SG.LGD, EL_CI.Normal.SG.LGD);
[AddOn.Normal.AR.LGD, AddOn_CI.Normal.AR.LGD] = Add_on(RC.Normal.AR.LGD,RC.naive.AR,EL.Normal.AR.LGD,EL.naive.AR,RC_CI.Normal.AR.LGD, EL_CI.Normal.AR.LGD);

% K simulation, fixing LGD 
tic
[~, EL.Normal.SG.K, RC.Normal.SG.K, RC_CI.Normal.SG.K, EL_CI.Normal.SG.K] = montecarlo_RC(N,k_hat_SG,sigma_SG,LGD_hat,sigma_LGD,alpha,g,M,2,corr_SG(1,2));
[~, EL.Normal.AR.K, RC.Normal.AR.K, RC_CI.Normal.AR.K, EL_CI.Normal.AR.K] = montecarlo_RC(N,k_hat_AR,sigma_AR,LGD_hat,sigma_LGD,alpha,g,M,2,corr_AR(1,2));
fprintf("K simulation - Montecarlo simulations computation time: ")
toc
% Add-on computation
[AddOn.Normal.SG.K, AddOn_CI.Normal.SG.K] = Add_on(RC.Normal.SG.K,RC.naive.SG,EL.Normal.SG.K,EL.naive.SG,RC_CI.Normal.SG.K, EL_CI.Normal.SG.K);
[AddOn.Normal.AR.K, AddOn_CI.Normal.AR.K] = Add_on(RC.Normal.AR.K,RC.naive.AR,EL.Normal.AR.K,EL.naive.AR,RC_CI.Normal.AR.K, EL_CI.Normal.AR.K);

% K and LGD independent
tic
[~, EL.Normal.SG.ind, RC.Normal.SG.ind, RC_CI.Normal.SG.ind, EL_CI.Normal.SG.ind] = montecarlo_RC(N,k_hat_SG,sigma_SG,LGD_hat,sigma_LGD,alpha,g,M,3,corr_SG(1,2));
[~, EL.Normal.AR.ind, RC.Normal.AR.ind, RC_CI.Normal.AR.ind, EL_CI.Normal.AR.ind] = montecarlo_RC(N,k_hat_AR,sigma_AR,LGD_hat,sigma_LGD,alpha,g,M,3,corr_AR(1,2));
fprintf("K simulation - Montecarlo simulations computation time: ")
toc
% Add-on computation
[AddOn.Normal.SG.ind, AddOn_CI.Normal.SG.ind] = Add_on(RC.Normal.SG.ind,RC.naive.SG,EL.Normal.SG.ind,EL.naive.SG,RC_CI.Normal.SG.ind, EL_CI.Normal.SG.ind);
[AddOn.Normal.AR.ind, AddOn_CI.Normal.AR.ind] = Add_on(RC.Normal.AR.ind,RC.naive.AR,EL.Normal.AR.ind,EL.naive.AR,RC_CI.Normal.AR.ind, EL_CI.Normal.AR.ind);

% K and LGD dependent
tic
[~, EL.Normal.SG.dep, RC.Normal.SG.dep, RC_CI.Normal.SG.dep, EL_CI.Normal.SG.dep] = montecarlo_RC(N,k_hat_SG,sigma_SG,LGD_hat,sigma_LGD,alpha,g,M,4,corr_SG(1,2));
[~, EL.Normal.AR.dep, RC.Normal.AR.dep, RC_CI.Normal.AR.dep, EL_CI.Normal.AR.dep] = montecarlo_RC(N,k_hat_AR,sigma_AR,LGD_hat,sigma_LGD,alpha,g,M,4,corr_AR(1,2));
fprintf("K simulation - Montecarlo simulations computation time: ")
toc
% Add-on computation
[AddOn.Normal.SG.dep, AddOn_CI.Normal.SG.dep] = Add_on(RC.Normal.SG.dep,RC.naive.SG,EL.Normal.SG.dep,EL.naive.SG,RC_CI.Normal.SG.dep, EL_CI.Normal.SG.dep);
[AddOn.Normal.AR.dep, AddOn_CI.Normal.AR.dep] = Add_on(RC.Normal.AR.dep,RC.naive.AR,EL.Normal.AR.dep,EL.naive.AR,RC_CI.Normal.AR.dep, EL_CI.Normal.AR.dep);

for i=1:length(alpha)
fprintf("\n")
fprintf("Add-on (alpha = %.3f)\n\n",alpha(i))
fprintf("Only LGD simulation:              <strong>SG</strong> = %.4f %% - IC = [%.4f,%.4f] %%      <strong>AR</strong> = %.4f %% - IC = [%.4f,%.4f] %% \n",AddOn.Normal.SG.LGD(i)*100,AddOn_CI.Normal.SG.LGD(i,1)*100,AddOn_CI.Normal.SG.LGD(i,2)*100,AddOn.Normal.AR.LGD(i)*100,AddOn_CI.Normal.AR.LGD(i,1)*100,AddOn_CI.Normal.AR.LGD(i,2)*100)
fprintf("Only K simulation:                <strong>SG</strong> = %.4f %% - IC = [%.4f,%.4f] %%      <strong>AR</strong> = %.4f %% - IC = [%.4f,%.4f] %% \n",AddOn.Normal.SG.K(i)*100,AddOn_CI.Normal.SG.K(i,1)*100,AddOn_CI.Normal.SG.K(i,2)*100,AddOn.Normal.AR.K(i)*100,AddOn_CI.Normal.AR.K(i,1)*100,AddOn_CI.Normal.AR.K(i,2)*100)
fprintf("LGD - K independent simulation:   <strong>SG</strong> = %.4f %% - IC = [%.4f,%.4f] %%      <strong>AR</strong> = %.4f %% - IC = [%.4f,%.4f] %% \n",AddOn.Normal.SG.ind(i)*100,AddOn_CI.Normal.SG.ind(i,1)*100,AddOn_CI.Normal.SG.ind(i,2)*100,AddOn.Normal.AR.ind(i)*100,AddOn_CI.Normal.AR.ind(i,1)*100,AddOn_CI.Normal.AR.ind(i,2)*100)
fprintf("LGD - K dependent simulation:     <strong>SG</strong> = %.4f %% - IC = [%.4f,%.4f] %%      <strong>AR</strong> = %.4f %% - IC = [%.4f,%.4f] %% \n\n",AddOn.Normal.SG.dep(i)*100,AddOn_CI.Normal.SG.dep(i,1)*100,AddOn_CI.Normal.SG.dep(i,2)*100,AddOn.Normal.AR.dep(i)*100,AddOn_CI.Normal.AR.dep(i,1)*100,AddOn_CI.Normal.AR.dep(i,2)*100)
end

%% Homogeneous portfolio with various obligors
N_obligors = [50, 100, 250, 500, 1000]; % Number of obligors
N1 = 3*1e5; % Number of simulations for each issuer

%% Market parameters simulation
rng(seed_m)
M1 = randn(1,N1);

%% k and LGD standard normal simulation
rng(seed)
g1 = randn(2,N1); % k and LGD standard gaussian simulation

%% Montecarlo simulation
% We consider both alpha and all the obligors in order to compare them with 
% the same parameters
fprintf("\n<strong>------------------ Montecarlo Regulatory capital HP Stress - Add on computation -----------------</strong>\n\n")

for j=1:length(N_obligors)
    rng(seed)
    
    % Idiosinchratic terms simulation
    epsilon = randn(N_obligors(j),N1); % idiosinchratic terms simulation
    tic
    % Fix K, only LGD simulation
    % Montecarlo
    [~, EL.HP.SG.LGD(j,:), RC.HP.SG.LGD(j,:), RC_CI.HP.SG.LGD(j,:,:), EL_CI.HP.SG.LGD(j,:,:)] = montecarlo_RC_HP(N1,norminv(PD_SG_hat),sigma_SG,LGD_hat,sigma_LGD,alpha,g1,M1,epsilon,1,N_obligors(j),corr_SG);
    [~, EL.HP.AR.LGD(j,:), RC.HP.AR.LGD(j,:), RC_CI.HP.AR.LGD(j,:,:), EL_CI.HP.AR.LGD(j,:,:)] = montecarlo_RC_HP(N1,norminv(PD_AR_hat),sigma_AR,LGD_hat,sigma_LGD,alpha,g1,M1,epsilon,1,N_obligors(j),corr_AR);
    % Add-on
    [AddOn.HP.SG.LGD(j,:), AddOn_CI.HP.SG.LGD(j,:,:)] = Add_on(RC.HP.SG.LGD(j,:),RC.naive.SG,EL.HP.SG.LGD(j,:),EL.naive.SG,squeeze(RC_CI.HP.SG.LGD(j,:,:)), squeeze(EL_CI.HP.SG.LGD(j,:,:)));
    [AddOn.HP.AR.LGD(j,:), AddOn_CI.HP.AR.LGD(j,:,:)] = Add_on(RC.HP.AR.LGD(j,:),RC.naive.AR,EL.HP.AR.LGD(j,:),EL.naive.AR,squeeze( RC_CI.HP.AR.LGD(j,:,:)), squeeze(EL_CI.HP.AR.LGD(j,:,:)));
    
    % Fix LGD, only K simulation
    % Montecarlo
    [~, EL.HP.SG.K(j,:), RC.HP.SG.K(j,:), RC_CI.HP.SG.K(j,:,:), EL_CI.HP.SG.K(j,:,:)] = montecarlo_RC_HP(N1,k_hat_SG,sigma_SG,LGD_hat,sigma_LGD,alpha,g1,M1,epsilon,2,N_obligors(j));
    [~, EL.HP.AR.K(j,:), RC.HP.AR.K(j,:), RC_CI.HP.AR.K(j,:,:), EL_CI.HP.AR.K(j,:,:)] = montecarlo_RC_HP(N1,k_hat_AR,sigma_AR,LGD_hat,sigma_LGD,alpha,g1,M1,epsilon,2,N_obligors(j));
    % Add-on
    [AddOn.HP.SG.K(j,:), AddOn_CI.HP.SG.K(j,:,:)] = Add_on(RC.HP.SG.K(j,:),RC.naive.SG,EL.HP.SG.K(j,:),EL.naive.SG,squeeze(RC_CI.HP.SG.K(j,:,:)), squeeze(EL_CI.HP.SG.K(j,:,:)));
    [AddOn.HP.AR.K(j,:), AddOn_CI.HP.AR.K(j,:,:)] = Add_on(RC.HP.AR.K(j,:),RC.naive.AR,EL.HP.AR.K(j,:),EL.naive.AR,squeeze(RC_CI.HP.AR.K(j,:,:)), squeeze(EL_CI.HP.AR.K(j,:,:)));
    
    % Independent case
    % Montecarlo
    [~, EL.HP.SG.ind(j,:), RC.HP.SG.ind(j,:), RC_CI.HP.SG.ind(j,:,:), EL_CI.HP.SG.ind(j,:,:)] = montecarlo_RC_HP(N1,k_hat_SG,sigma_SG,LGD_hat,sigma_LGD,alpha,g1,M1,epsilon,3,N_obligors(j));
    [~, EL.HP.AR.ind(j,:), RC.HP.AR.ind(j,:), RC_CI.HP.AR.ind(j,:,:), EL_CI.HP.AR.ind(j,:,:)] = montecarlo_RC_HP(N1,k_hat_AR,sigma_AR,LGD_hat,sigma_LGD,alpha,g1,M1,epsilon,3,N_obligors(j));
    % Add-on
    [AddOn.HP.SG.ind(j,:), AddOn_CI.HP.SG.ind(j,:,:)] = Add_on(RC.HP.SG.ind(j,:),RC.naive.SG,EL.HP.SG.ind(j,:),EL.naive.SG,squeeze(RC_CI.HP.SG.ind(j,:,:)), squeeze(EL_CI.HP.SG.ind(j,:,:)));
    [AddOn.HP.AR.ind(j,:), AddOn_CI.HP.AR.ind(j,:,:)] = Add_on(RC.HP.AR.ind(j,:),RC.naive.AR,EL.HP.AR.ind(j,:),EL.naive.AR,squeeze(RC_CI.HP.AR.ind(j,:,:)), squeeze(EL_CI.HP.AR.ind(j,:,:)));
    
    % Dependent case
    % Montecarlo
    [~, EL.HP.SG.dep(j,:), RC.HP.SG.dep(j,:), RC_CI.HP.SG.dep(j,:,:), EL_CI.HP.SG.dep(j,:,:)] = montecarlo_RC_HP(N1,k_hat_SG,sigma_SG,LGD_hat,sigma_LGD,alpha,g1,M1,epsilon,4,N_obligors(j),corr_SG);
    [~, EL.HP.AR.dep(j,:), RC.HP.AR.dep(j,:), RC_CI.HP.AR.dep(j,:,:), EL_CI.HP.AR.dep(j,:,:)] = montecarlo_RC_HP(N1,k_hat_AR,sigma_AR,LGD_hat,sigma_LGD,alpha,g1,M1,epsilon,4,N_obligors(j),corr_AR);
    % Add-on
    [AddOn.HP.SG.dep(j,:), AddOn_CI.HP.SG.dep(j,:,:)] = Add_on(RC.HP.SG.dep(j,:),RC.naive.SG,EL.HP.SG.dep(j,:),EL.naive.SG,squeeze(RC_CI.HP.SG.dep(j,:,:)), squeeze(EL_CI.HP.SG.dep(j,:,:)));
    [AddOn.HP.AR.dep(j,:), AddOn_CI.HP.AR.dep(j,:,:)] = Add_on(RC.HP.AR.dep(j,:),RC.naive.AR,EL.HP.AR.dep(j,:),EL.naive.AR,squeeze(RC_CI.HP.AR.dep(j,:,:)), squeeze(EL_CI.HP.AR.dep(j,:,:)));

    for i=1:length(alpha)
        fprintf("\n")
        fprintf("Add-on (alpha = %.3f - N obligors = %d)\n\n",alpha(i),N_obligors(j))
        fprintf("Only LGD simulation:                <strong>SG</strong> = %.4f %% - CI = [%.4f, %.4f] %% <strong>AR</strong> = %.4f %% - CI = [%.4f, %.4f] %% \n",AddOn.HP.SG.LGD(j,i)*100,AddOn_CI.HP.SG.LGD(j,i,1)*100,AddOn_CI.HP.SG.LGD(j,i,2)*100,AddOn.HP.AR.LGD(j,i)*100,AddOn_CI.HP.AR.LGD(j,i,1)*100,AddOn_CI.HP.AR.LGD(j,i,2)*100)
        fprintf("Only K simulation:                  <strong>SG</strong> = %.4f %% - CI = [%.4f, %.4f] %% <strong>AR</strong> = %.4f %% - CI = [%.4f, %.4f] %% \n",AddOn.HP.SG.K(j,i)*100,AddOn_CI.HP.SG.K(j,i,1)*100,AddOn_CI.HP.SG.K(j,i,2)*100,AddOn.HP.AR.K(j,i)*100,AddOn_CI.HP.AR.K(j,i,1)*100,AddOn_CI.HP.AR.K(j,i,2)*100)
        fprintf("LGD - K independent simulation:     <strong>SG</strong> = %.4f %% - CI = [%.4f, %.4f] %% <strong>AR</strong> = %.4f %% - CI = [%.4f, %.4f] %% \n",AddOn.HP.SG.ind(j,i)*100,AddOn_CI.HP.SG.ind(j,i,1)*100,AddOn_CI.HP.SG.ind(j,i,2)*100,AddOn.HP.AR.ind(j,i)*100,AddOn_CI.HP.AR.ind(j,i,1)*100,AddOn_CI.HP.AR.ind(j,i,2)*100)
        fprintf("LGD - K dependent simulation:       <strong>SG</strong> = %.4f %% - CI = [%.4f, %.4f] %%   <strong>AR</strong> = %.4f %% - CI = [%.4f, %.4f] %% \n\n",AddOn.HP.SG.dep(j,i)*100,AddOn_CI.HP.SG.dep(j,i,1)*100,AddOn_CI.HP.SG.dep(j,i,2)*100,AddOn.HP.AR.dep(j,i)*100,AddOn_CI.HP.AR.dep(j,i,1)*100,AddOn_CI.HP.AR.dep(j,i,2)*100)
    end
    fprintf("Montecarlo simulations computation time (N obligors = %d): ",N_obligors(j))
    toc
end

%% Plots
for i = 1:length(alpha)
    figure('Name','HP convergence plot')
    
    subplot(4,2,1)
    plot(N_obligors,RC.HP.AR.LGD(:,i),'LineWidth',2)
    hold on
    plot(N_obligors,RC.Normal.AR.LGD(i)*ones(size(N_obligors)),'LineWidth',2)
    hold off
    title('Simulating LGD only - AR case')
    xlabel('obligors')
    ylabel('RC')
    
    subplot(4,2,2)
    plot(N_obligors,RC.HP.SG.LGD(:,i),'LineWidth',2)
    hold on
    plot(N_obligors,RC.Normal.SG.LGD(i)*ones(size(N_obligors)),'LineWidth',2)
    hold off
    title('Simulating LGD only - SG case')
    xlabel('obligors')
    ylabel('RC')
    
    subplot(4,2,3)
    plot(N_obligors,RC.HP.AR.K(:,i),'LineWidth',2)
    hold on
    plot(N_obligors,RC.Normal.AR.K(i)*ones(size(N_obligors)),'LineWidth',2)
    hold off
    title('Simulating K only - AR case')
    xlabel('obligors')
    ylabel('RC')
    
    subplot(4,2,4)
    plot(N_obligors,RC.HP.SG.K(:,i),'LineWidth',2)
    hold on
    plot(N_obligors,RC.Normal.SG.K(i)*ones(size(N_obligors)),'LineWidth',2)
    hold off
    title('Simulating K only - SG case')
    xlabel('obligors')
    ylabel('RC')
    
    subplot(4,2,5)
    plot(N_obligors,RC.HP.AR.ind(:,i),'LineWidth',2)
    hold on
    plot(N_obligors,RC.Normal.AR.ind(i)*ones(size(N_obligors)),'LineWidth',2)
    hold off
    title('Independent - AR case')
    xlabel('obligors')
    ylabel('RC')
    
    subplot(4,2,6)
    plot(N_obligors,RC.HP.SG.ind(:,i),'LineWidth',2)
    hold on
    plot(N_obligors,RC.Normal.SG.ind(i)*ones(size(N_obligors)),'LineWidth',2)
    hold off
    title('Independent  - SG case')
    xlabel('obligors')
    ylabel('RC')
    
    subplot(4,2,7)
    plot(N_obligors,RC.HP.AR.dep(:,i),'LineWidth',2)
    hold on
    plot(N_obligors,RC.Normal.AR.dep(i)*ones(size(N_obligors)),'LineWidth',2)
    hold off
    title('Dependent - AR case')
    xlabel('obligors')
    ylabel('RC')
    
    subplot(4,2,8)
    plot(N_obligors,RC.HP.SG.dep(:,i),'LineWidth',2)
    hold on
    plot(N_obligors,RC.Normal.SG.dep(i)*ones(size(N_obligors)),'LineWidth',2)
    hold off
    title('Dependent - SG case')
    xlabel('obligors')
    ylabel('RC')

    txt=['HP convergence plot, alpha=',num2str(alpha(i))];
    sgtitle(txt)
end

%% k and LGD modelled as a t-student (we consider only 99.9% confidence level)

fprintf('\n <strong>--------------- t-STUDENT CASE ---------------</strong>\n\n');

for i=1:19 % from dof = 2 to dof = 20
    tic
    dof = i+1; % degrees of freedom

    % LGD and K t-Student simulation
    rng(seed)
    t = trnd(dof,2,N);

    % LGD simulation, fixing K     
    [~, EL.T.SG.LGD, RC.T.SG.LGD(i), RC_CI.T.SG.LGD(i,:), EL_CI.T.SG.LGD] = montecarlo_RC_T(N,norminv(PD_SG_hat),sigma_SG,LGD_hat,sigma_LGD,alpha(1),t,M,1,corr_SG(1,2));
    [~, EL.T.AR.LGD, RC.T.AR.LGD(i), RC_CI.T.AR.LGD(i,:), EL_CI.T.AR.LGD] = montecarlo_RC_T(N,norminv(PD_AR_hat),sigma_AR,LGD_hat,sigma_LGD,alpha(1),t,M,1,corr_AR(1,2));
    % Add-on computation
    [AddOn.T.SG.LGD(i), AddOn_CI.T.SG.LGD(i,:)] = Add_on(RC.T.SG.LGD(i),RC.naive.SG(1),EL.T.SG.LGD,EL.naive.SG,RC_CI.T.SG.LGD(i,:), EL_CI.T.SG.LGD);
    [AddOn.T.AR.LGD(i), AddOn_CI.T.AR.LGD(i,:)] = Add_on(RC.T.AR.LGD(i),RC.naive.AR(1),EL.T.AR.LGD,EL.naive.AR,RC_CI.T.AR.LGD(i,:), EL_CI.T.AR.LGD);
    
    % K simulation, fixing LGD 
    [~, EL.T.SG.K, RC.T.SG.K(i), RC_CI.T.SG.K(i,:), EL_CI.T.SG.K] = montecarlo_RC_T(N,k_hat_SG,sigma_SG,LGD_hat,sigma_LGD,alpha(1),t,M,2,corr_SG(1,2));
    [~, EL.T.AR.K, RC.T.AR.K(i), RC_CI.T.AR.K(i,:), EL_CI.T.AR.K] = montecarlo_RC_T(N,k_hat_AR,sigma_AR,LGD_hat,sigma_LGD,alpha(1),t,M,2,corr_AR(1,2));
    % Add-on computation
    [AddOn.T.SG.K(i), AddOn_CI.T.SG.K(i,:)] = Add_on(RC.T.SG.K(i),RC.naive.SG(1),EL.T.SG.K,EL.naive.SG,RC_CI.T.SG.K(i,:), EL_CI.T.SG.K);
    [AddOn.T.AR.K(i), AddOn_CI.T.AR.K(i,:)] = Add_on(RC.T.AR.K(i),RC.naive.AR(1),EL.T.AR.K,EL.naive.AR,RC_CI.T.AR.K(i,:), EL_CI.T.AR.K);

    % K and LGD independent
    [~, EL.T.SG.ind, RC.T.SG.ind(i), RC_CI.T.SG.ind(i,:), EL_CI.T.SG.ind] = montecarlo_RC_T(N,k_hat_SG,sigma_SG,LGD_hat,sigma_LGD,alpha(1),t,M,3,corr_SG(1,2));
    [~, EL.T.AR.ind, RC.T.AR.ind(i), RC_CI.T.AR.ind(i,:), EL_CI.T.AR.ind] = montecarlo_RC_T(N,k_hat_AR,sigma_AR,LGD_hat,sigma_LGD,alpha(1),t,M,3,corr_AR(1,2));
    % Add-on computation
    [AddOn.T.SG.ind(i), AddOn_CI.T.SG.ind(i,:)] = Add_on(RC.T.SG.ind(i),RC.naive.SG(1),EL.T.SG.ind,EL.naive.SG,RC_CI.T.SG.ind(i,:), EL_CI.T.SG.ind);
    [AddOn.T.AR.ind(i), AddOn_CI.T.AR.ind(i,:)] = Add_on(RC.T.AR.ind(i),RC.naive.AR(1),EL.T.AR.ind,EL.naive.AR,RC_CI.T.AR.ind(i,:), EL_CI.T.AR.ind);
    
    % K and LGD dependent
    [~, EL.T.SG.dep, RC.T.SG.dep(i), RC_CI.T.SG.dep(i,:), EL_CI.T.SG.dep] = montecarlo_RC_T(N,k_hat_SG,sigma_SG,LGD_hat,sigma_LGD,alpha(1),t,M,4,corr_SG(1,2));
    [~, EL.T.AR.dep, RC.T.AR.dep(i), RC_CI.T.AR.dep(i,:), EL_CI.T.AR.dep] = montecarlo_RC_T(N,k_hat_AR,sigma_AR,LGD_hat,sigma_LGD,alpha(1),t,M,4,corr_AR(1,2));  
    % Add-on computation
    [AddOn.T.SG.dep(i), AddOn_CI.T.SG.dep(i,:)] = Add_on(RC.T.SG.dep(i),RC.naive.SG(1),EL.T.SG.dep,EL.naive.SG,RC_CI.T.SG.dep(i,:), EL_CI.T.SG.dep);
    [AddOn.T.AR.dep(i), AddOn_CI.T.AR.dep(i,:)] = Add_on(RC.T.AR.dep(i),RC.naive.AR(1),EL.T.AR.dep,EL.naive.AR,RC_CI.T.AR.dep(i,:), EL_CI.T.AR.dep);

    fprintf("\n")
    fprintf("Add-on (alpha = %.3f - dof = %d)\n\n",alpha(1),dof)
    fprintf("Only LGD simulation:                <strong>SG</strong> = %.4f %% - CI = [%.4f, %.4f] %% <strong>AR</strong> = %.4f %% - CI = [%.4f, %.4f] %% \n",AddOn.T.SG.LGD(i)*100,AddOn_CI.T.SG.LGD(i,1)*100,AddOn_CI.T.SG.LGD(i,2)*100,AddOn.T.AR.LGD(i)*100,AddOn_CI.T.AR.LGD(i,1)*100,AddOn_CI.T.AR.LGD(i,2)*100)
    fprintf("Only K simulation:                  <strong>SG</strong> = %.4f %% - CI = [%.4f, %.4f] %%  <strong>AR</strong> = %.4f %% - CI = [%.4f, %.4f] %% \n",AddOn.T.SG.K(i)*100,AddOn_CI.T.SG.K(i,1)*100,AddOn_CI.T.SG.K(i,2)*100,AddOn.T.AR.K(i)*100,AddOn_CI.T.AR.K(i,1)*100,AddOn_CI.T.AR.K(i,2)*100)
    fprintf("LGD - K independent simulation:     <strong>SG</strong> = %.4f %% - CI = [%.4f, %.4f] %%  <strong>AR</strong> = %.4f %% - CI = [%.4f, %.4f] %% \n",AddOn.T.SG.ind(i)*100,AddOn_CI.T.SG.ind(i,1)*100,AddOn_CI.T.SG.ind(i,2)*100,AddOn.T.AR.ind(i)*100,AddOn_CI.T.AR.ind(i,1)*100,AddOn_CI.T.AR.ind(i,2)*100)
    fprintf("LGD - K dependent simulation:       <strong>SG</strong> = %.4f %% - CI = [%.4f, %.4f] %%  <strong>AR</strong> = %.4f %% - CI = [%.4f, %.4f] %% \n\n",AddOn.T.SG.dep(i)*100,AddOn_CI.T.SG.dep(i,1)*100,AddOn_CI.T.SG.dep(i,2)*100,AddOn.T.AR.dep(i)*100,AddOn_CI.T.AR.dep(i,1)*100,AddOn_CI.T.AR.dep(i,2)*100)
    fprintf("Montecarlo simulations computation time, for all cases, with t-Student (dof = %d):   ", dof)
    toc
end

%% plot t-student convergence

x = 2:20;

figure('Name','t-Student convergence plot')
sgtitle('t-Student convergence plot, alpha=0.999')

subplot(4,2,1)
plot(x,RC.T.AR.LGD,'LineWidth',2)
hold on
plot(x,RC.Normal.AR.LGD(1)*ones(19),'LineWidth',2)
hold off
title('Simulating LGD only - AR case')
xlabel('degrees of freedom')
ylabel('RC')

subplot(4,2,2)
plot(x,RC.T.SG.LGD,'LineWidth',2)
hold on
plot(x,RC.Normal.SG.LGD(1)*ones(19),'LineWidth',2)
hold off
title('Simulating LGD only - SG case')
xlabel('degrees of freedom')
ylabel('RC')

subplot(4,2,3)
plot(x,RC.T.AR.K,'LineWidth',2)
hold on
plot(x,RC.Normal.AR.K(1)*ones(19),'LineWidth',2)
hold off
title('Simulating K only - AR case')
xlabel('degrees of freedom')
ylabel('RC')

subplot(4,2,4)
plot(x,RC.T.SG.K,'LineWidth',2)
hold on
plot(x,RC.Normal.SG.K(1)*ones(19),'LineWidth',2)
hold off
title('Simulating K only - SG case')
xlabel('degrees of freedom')
ylabel('RC')

subplot(4,2,5)
plot(x,RC.T.AR.ind,'LineWidth',2)
hold on
plot(x,RC.Normal.AR.ind(1)*ones(19),'LineWidth',2)
hold off
title('Simulating LGD and K as independent - AR case')
xlabel('degrees of freedom')
ylabel('RC')

subplot(4,2,6)
plot(x,RC.T.SG.ind,'LineWidth',2)
hold on
plot(x,RC.Normal.SG.ind(1)*ones(19),'LineWidth',2)
hold off
title('Simulating LGD and K as independent - SG case')
xlabel('degrees of freedom')
ylabel('RC')

subplot(4,2,7)
plot(x,RC.T.AR.dep,'LineWidth',2)
hold on
plot(x,RC.Normal.AR.dep(1)*ones(19),'LineWidth',2)
hold off
title('Simulating LGD and K as dependent - AR case')
xlabel('degrees of freedom')
ylabel('RC')

subplot(4,2,8)
plot(x,RC.T.SG.dep,'LineWidth',2)
hold on
plot(x,RC.Normal.SG.dep(1)*ones(19),'LineWidth',2)
hold off
title('Simulating LGD and K as dependent - SG case')
xlabel('degrees of freedom')
ylabel('RC')


%% Double t-student

fprintf('\n <strong>--------------- DOUBLE t-STUDENT CASE ---------------</strong>\n\n');

% Montecarlo
for dof=2:20
    tic
    rng(seed)

    % Double t-student parameters
    t1 = trnd(dof,2,N);
    t2 = trnd(dof,2,N);

    % LGD simulation, fixing K
    [~, EL.doubleT.SG.LGD(:,dof-1), RC.doubleT.SG.LGD(:,dof-1), RC_CI.doubleT.SG.LGD(:,dof-1,:), EL_CI.doubleT.SG.LGD(:,dof-1,:)] = montecarlo_RC_T_double(N,norminv(PD_SG_hat),sigma_SG,LGD_hat,sigma_LGD,alpha,t1,t2,M,1,corr_SG(1,2));
    [~, EL.doubleT.AR.LGD(:,dof-1), RC.doubleT.AR.LGD(:,dof-1), RC_CI.doubleT.AR.LGD(:,dof-1,:), EL_CI.doubleT.AR.LGD(:,dof-1,:)] = montecarlo_RC_T_double(N,norminv(PD_AR_hat),sigma_AR,LGD_hat,sigma_LGD,alpha,t1,t2,M,1,corr_AR(1,2));
    % Add-on computation
    [AddOn.doubleT.SG.LGD(:,dof-1), AddOn_CI.doubleT.SG.LGD(:,dof-1,:)] = Add_on(RC.doubleT.SG.LGD(:,dof-1)',RC.naive.SG,EL.doubleT.SG.LGD(:,dof-1),EL.naive.SG,squeeze(RC_CI.doubleT.SG.LGD(:,dof-1,:)), squeeze(EL_CI.doubleT.SG.LGD(:,dof-1,:)));
    [AddOn.doubleT.AR.LGD(:,dof-1), AddOn_CI.doubleT.AR.LGD(:,dof-1,:)] = Add_on(RC.doubleT.AR.LGD(:,dof-1)',RC.naive.AR,EL.doubleT.AR.LGD(:,dof-1),EL.naive.AR,squeeze(RC_CI.doubleT.AR.LGD(:,dof-1,:)), squeeze(EL_CI.doubleT.AR.LGD(:,dof-1,:)));
    
    % K simulation, fixing LGD     
    [~, EL.doubleT.SG.K(:,dof-1), RC.doubleT.SG.K(:,dof-1), RC_CI.doubleT.SG.K(:,dof-1,:), EL_CI.doubleT.SG.K(:,dof-1,:)] = montecarlo_RC_T_double(N,k_hat_SG,sigma_SG,LGD_hat,sigma_LGD,alpha,t1,t2,M,2,corr_SG(1,2));
    [~, EL.doubleT.AR.K(:,dof-1), RC.doubleT.AR.K(:,dof-1), RC_CI.doubleT.AR.K(:,dof-1,:), EL_CI.doubleT.AR.K(:,dof-1,:)] = montecarlo_RC_T_double(N,k_hat_AR,sigma_AR,LGD_hat,sigma_LGD,alpha,t1,t2,M,2,corr_AR(1,2));
    % Add-on computation
    [AddOn.doubleT.SG.K(:,dof-1), AddOn_CI.doubleT.SG.K(:,dof-1,:)] = Add_on(RC.doubleT.SG.K(:,dof-1)',RC.naive.SG,EL.doubleT.SG.K(:,dof-1),EL.naive.SG,squeeze(RC_CI.doubleT.SG.K(:,dof-1,:)), squeeze(EL_CI.doubleT.SG.K(:,dof-1,:)));
    [AddOn.doubleT.AR.K(:,dof-1), AddOn_CI.doubleT.AR.K(:,dof-1,:)] = Add_on(RC.doubleT.AR.K(:,dof-1)',RC.naive.AR,EL.doubleT.AR.K(:,dof-1),EL.naive.AR,squeeze(RC_CI.doubleT.AR.K(:,dof-1,:)), squeeze(EL_CI.doubleT.AR.K(:,dof-1,:)));

    % K and LGD independent
    [~, EL.doubleT.SG.ind(:,dof-1), RC.doubleT.SG.ind(:,dof-1), RC_CI.doubleT.SG.ind(:,dof-1,:), EL_CI.doubleT.SG.ind(:,dof-1,:)] = montecarlo_RC_T_double(N,k_hat_SG,sigma_SG,LGD_hat,sigma_LGD,alpha,t1,t2,M,3,corr_SG(1,2));
    [~, EL.doubleT.AR.ind(:,dof-1), RC.doubleT.AR.ind(:,dof-1), RC_CI.doubleT.AR.ind(:,dof-1,:), EL_CI.doubleT.AR.ind(:,dof-1,:)] = montecarlo_RC_T_double(N,k_hat_AR,sigma_AR,LGD_hat,sigma_LGD,alpha,t1,t2,M,3,corr_AR(1,2));
    % Add-on computation
    [AddOn.doubleT.SG.ind(:,dof-1), AddOn_CI.doubleT.SG.ind(:,dof-1,:)] = Add_on(RC.doubleT.SG.ind(:,dof-1)',RC.naive.SG,EL.doubleT.SG.ind(:,dof-1),EL.naive.SG,squeeze(RC_CI.doubleT.SG.ind(:,dof-1,:)), squeeze(EL_CI.doubleT.SG.ind(:,dof-1,:)));
    [AddOn.doubleT.AR.ind(:,dof-1), AddOn_CI.doubleT.AR.ind(:,dof-1,:)] = Add_on(RC.doubleT.AR.ind(:,dof-1)',RC.naive.AR,EL.doubleT.AR.ind(:,dof-1),EL.naive.AR,squeeze(RC_CI.doubleT.AR.ind(:,dof-1,:)), squeeze(EL_CI.doubleT.AR.ind(:,dof-1,:)));
    
    % K and LGD dependent
    [~, EL.doubleT.SG.dep(:,dof-1), RC.doubleT.SG.dep(:,dof-1), RC_CI.doubleT.SG.dep(:,dof-1,:), EL_CI.doubleT.SG.dep(:,dof-1,:)] = montecarlo_RC_T_double(N,k_hat_SG,sigma_SG,LGD_hat,sigma_LGD,alpha,t1,t2,M,4,corr_SG(1,2));
    [~, EL.doubleT.AR.dep(:,dof-1), RC.doubleT.AR.dep(:,dof-1), RC_CI.doubleT.AR.dep(:,dof-1,:), EL_CI.doubleT.AR.dep(:,dof-1,:)] = montecarlo_RC_T_double(N,k_hat_AR,sigma_AR,LGD_hat,sigma_LGD,alpha,t1,t2,M,4,corr_AR(1,2));       
    % Add-on computation
    [AddOn.doubleT.SG.dep(:,dof-1), AddOn_CI.doubleT.SG.dep(:,dof-1,:)] = Add_on(RC.doubleT.SG.dep(:,dof-1)',RC.naive.SG,EL.doubleT.SG.dep(:,dof-1),EL.naive.SG,squeeze(RC_CI.doubleT.SG.dep(:,dof-1,:)), squeeze(EL_CI.doubleT.SG.dep(:,dof-1,:)));
    [AddOn.doubleT.AR.dep(:,dof-1), AddOn_CI.doubleT.AR.dep(:,dof-1,:)] = Add_on(RC.doubleT.AR.dep(:,dof-1)',RC.naive.AR,EL.doubleT.AR.dep(:,dof-1),EL.naive.AR,squeeze(RC_CI.doubleT.AR.dep(:,dof-1,:)), squeeze(EL_CI.doubleT.AR.dep(:,dof-1,:)));
    
    for i=1:length(alpha)
        fprintf("\n")
        fprintf("Add-on (alpha = %.3f - dof = %d)\n\n",alpha(i),dof)
        fprintf("Only LGD simulation:                <strong>SG</strong> = %.4f %% - CI = [%.4f, %.4f] %% <strong>AR</strong> = %.4f  %%- CI = [%.4f, %.4f] %% \n",AddOn.doubleT.SG.LGD(i,dof-1),AddOn_CI.doubleT.SG.LGD(i,dof-1,1),AddOn_CI.doubleT.SG.LGD(i,dof-1,2),AddOn.doubleT.AR.LGD(i,dof-1),AddOn_CI.doubleT.AR.LGD(i,dof-1,1),AddOn_CI.doubleT.AR.LGD(i,dof-1,2))
        fprintf("Only K simulation:                  <strong>SG</strong> = %.4f %% - CI = [%.4f, %.4f]  %% <strong>AR</strong> = %.4f %% - CI = [%.4f, %.4f] %% \n",AddOn.doubleT.SG.K(i,dof-1),AddOn_CI.doubleT.SG.K(i,dof-1,1),AddOn_CI.doubleT.SG.K(i,dof-1,2),AddOn.doubleT.AR.K(i,dof-1),AddOn_CI.doubleT.AR.K(i,dof-1,1),AddOn_CI.doubleT.AR.K(i,dof-1,2))
        fprintf("LGD - K independent simulation:     <strong>SG</strong> = %.4f %% - CI = [%.4f, %.4f] %%  <strong>AR</strong> = %.4f %% - CI = [%.4f, %.4f] %% \n",AddOn.doubleT.SG.ind(i,dof-1),AddOn_CI.doubleT.SG.ind(i,dof-1,1),AddOn_CI.doubleT.SG.ind(i,dof-1,2),AddOn.doubleT.AR.ind(i,dof-1),AddOn_CI.doubleT.AR.ind(i,dof-1,1),AddOn_CI.doubleT.AR.ind(i,dof-1,2))
        fprintf("LGD - K dependent simulation:       <strong>SG</strong> = %.4f %% - CI = [%.4f, %.4f] %%  <strong>AR</strong> = %.4f %% - CI = [%.4f, %.4f] %% \n\n",AddOn.doubleT.SG.dep(i,dof-1),AddOn_CI.doubleT.SG.dep(i,dof-1,1),AddOn_CI.doubleT.SG.dep(i,dof-1,2),AddOn.doubleT.AR.dep(i,dof-1),AddOn_CI.doubleT.AR.dep(i,dof-1,1),AddOn_CI.doubleT.AR.dep(i,dof-1,2))
    end
    fprintf("Montecarlo simulations computation time (dof = %d): ",dof)
    toc
end

%% Plot double t-student convergence

x = 2:20;

for i = 1:length(alpha)
    figure('Name','double t-Student convergence plot')

    subplot(4,2,1)
    plot(x,RC.doubleT.AR.LGD(i,:),'LineWidth',2)
    hold on
    plot(x,RC.Normal.AR.LGD(i)*ones(length(x),1),'LineWidth',2)
    hold off
    title('Simulating LGD only - AR case')
    xlabel('degree of freedom')
    ylabel('RC')
    
    subplot(4,2,2)
    plot(x,RC.doubleT.SG.LGD(i,:),'LineWidth',2)
    hold on
    plot(x,RC.Normal.SG.LGD(i)*ones(length(x),1),'LineWidth',2)
    hold off
    title('Simulating LGD only - SG case')
    xlabel('degree of freedom')
    ylabel('RC')
    
    subplot(4,2,3)
    plot(x,RC.doubleT.AR.K(i,:),'LineWidth',2)
    hold on
    plot(x,RC.Normal.AR.K(i)*ones(length(x),1),'LineWidth',2)
    hold off
    title('Simulating K only - AR case')
    xlabel('degree of freedom')
    ylabel('RC')
    
    subplot(4,2,4)
    plot(x,RC.doubleT.SG.K(i,:),'LineWidth',2)
    hold on
    plot(x,RC.Normal.SG.K(i)*ones(length(x),1),'LineWidth',2)
    hold off
    title('Simulating K only - SG case')
    xlabel('degree of freedom')
    ylabel('RC')
    
    subplot(4,2,5)
    plot(x,RC.doubleT.AR.ind(i,:),'LineWidth',2)
    hold on
    plot(x,RC.Normal.AR.ind(i)*ones(length(x),1),'LineWidth',2)
    hold off
    title('Simulating LGD and K independent - AR case')
    xlabel('degree of freedom')
    ylabel('RC')
    
    subplot(4,2,6)
    plot(x,RC.doubleT.SG.ind(i,:),'LineWidth',2)
    hold on
    plot(x,RC.Normal.SG.ind(i)*ones(length(x),1),'LineWidth',2)
    hold off
    title('Simulating LGD and K independent - SG case')
    xlabel('degree of freedom')
    ylabel('RC')
    
    subplot(4,2,7)
    plot(x,RC.doubleT.SG.dep(i,:),'LineWidth',2)
    hold on
    plot(x,RC.Normal.SG.dep(i)*ones(length(x),1),'LineWidth',2)
    hold off
    title('Simulating LGD and K dependent - AR case')
    xlabel('degree of freedom')
    ylabel('RC')
    
    subplot(4,2,8)
    plot(x,RC.doubleT.SG.dep(i,:),'LineWidth',2)
    hold on
    plot(x,RC.Normal.SG.dep(i)*ones(length(x),1),'LineWidth',2)
    hold off
    title('Simulating LGD and K dependent - SG case')
    xlabel('degree of freedom')
    ylabel('RC')

    txt=['double t-Student convergence plot, alpha=',num2str(alpha(i))];
    sgtitle(txt)
end

%% Plot T-student and double t-student

figure('Name','doubleT and T comparison')
sgtitle('doubleT and T comparison')

subplot(4,2,1)
plot(x,RC.doubleT.AR.LGD(1,:),'LineWidth',2)
hold on
plot(x,RC.T.AR.LGD,'LineWidth',2)
hold off
title('Simulating LGD only - AR case')
xlabel('degree of freedom')
ylabel('RC')

subplot(4,2,2)
plot(x,RC.doubleT.SG.LGD(1,:),'LineWidth',2)
hold on
plot(x,RC.T.SG.LGD,'LineWidth',2)
hold off
title('Simulating LGD only - SG case')
xlabel('degree of freedom')
ylabel('RC')

subplot(4,2,3)
plot(x,RC.doubleT.AR.K(1,:),'LineWidth',2)
hold on
plot(x,RC.T.AR.K,'LineWidth',2)
hold off
title('Simulating K only - AR case')
xlabel('degree of freedom')
ylabel('RC')

subplot(4,2,4)
plot(x,RC.doubleT.SG.K(1,:),'LineWidth',2)
hold on
plot(x,RC.T.SG.K,'LineWidth',2)
hold off
title('Simulating K only - SG case')
xlabel('degree of freedom')
ylabel('RC')

subplot(4,2,5)
plot(x,RC.doubleT.AR.ind(1,:),'LineWidth',2)
hold on
plot(x,RC.T.AR.ind,'LineWidth',2)
hold off
title('Simulating LGD and K independent - AR case')
xlabel('degree of freedom')
ylabel('RC')

subplot(4,2,6)
plot(x,RC.doubleT.SG.ind(1,:),'LineWidth',2)
hold on
plot(x,RC.T.SG.ind,'LineWidth',2)
hold off
title('Simulating LGD and K independent - SG case')
xlabel('degree of freedom')
ylabel('RC')

subplot(4,2,7)
plot(x,RC.doubleT.SG.dep(1,:),'LineWidth',2)
hold on
plot(x,RC.T.SG.dep,'LineWidth',2)
hold off
title('Simulating LGD and K dependent - AR case')
xlabel('degree of freedom')
ylabel('RC')

subplot(4,2,8)
plot(x,RC.doubleT.SG.dep(1,:),'LineWidth',2)
hold on
plot(x,RC.T.SG.dep,'LineWidth',2)
hold off
title('Simulating LGD and K dependent - SG case')
xlabel('degree of freedom')
ylabel('RC')

figure('Name', 'doubleT and T comparison in a particular case')
plot(x,RC.doubleT.SG.K(1,:),'LineWidth',2)
hold on
plot(x,RC.T.SG.K,'LineWidth',2)
hold off
title('Simulating K only - SG case')
xlabel('degree of freedom')
ylabel('RC')
legend('double t','t')

%% Montecarlo simulation N = 50 double t-student
fprintf("\n<strong>------------------ Montecarlo Regulatory capital HP Stress Double t-student - Add on computation -----------------</strong>\n\n")

for dof=2:20
    rng(seed)
    tic
    
    % Idiosinchratic terms simulation
    epsilon = randn(N_obligors(1),N1); % idiosinchratic terms simulation
    t1_T = trnd(dof,2,N1);
    t2_T = trnd(dof,2,N1);
    
    % Fix K, only LGD simulation
    % Montecarlo
    [~, EL.HP_doubleT.SG.LGD(dof-1,:), RC.HP_doubleT.SG.LGD(dof-1,:), RC_CI.HP_doubleT.SG.LGD(dof-1,:,:), EL_CI.HP_doubleT.SG.LGD(dof-1,:,:)] = montecarlo_RC_HP_T_double(N1,norminv(PD_SG_hat),sigma_SG,LGD_hat,sigma_LGD,alpha,t1_T,t2_T,M1,epsilon,1,N_obligors(1));
    [~, EL.HP_doubleT.AR.LGD(dof-1,:), RC.HP_doubleT.AR.LGD(dof-1,:), RC_CI.HP_doubleT.AR.LGD(dof-1,:,:), EL_CI.HP_doubleT.AR.LGD(dof-1,:,:)] = montecarlo_RC_HP_T_double(N1,norminv(PD_AR_hat),sigma_AR,LGD_hat,sigma_LGD,alpha,t1_T,t2_T,M1,epsilon,1,N_obligors(1));
    % Add-on
    [AddOn.HP_doubleT.SG.LGD(dof-1,:), AddOn_CI.HP_doubleT.SG.LGD(dof-1,:,:)] = Add_on(RC.HP_doubleT.SG.LGD(dof-1,:),RC.naive.SG,EL.HP_doubleT.SG.LGD(dof-1,:),EL.naive.SG,squeeze(RC_CI.HP_doubleT.SG.LGD(dof-1,:,:)), squeeze(EL_CI.HP_doubleT.SG.LGD(dof-1,:,:)));
    [AddOn.HP_doubleT.AR.LGD(dof-1,:), AddOn_CI.HP_doubleT.AR.LGD(dof-1,:,:)] = Add_on(RC.HP_doubleT.AR.LGD(dof-1,:),RC.naive.AR,EL.HP_doubleT.AR.LGD(dof-1,:),EL.naive.AR,squeeze(RC_CI.HP_doubleT.AR.LGD(dof-1,:,:)), squeeze(EL_CI.HP_doubleT.AR.LGD(dof-1,:,:)));
    
    % Fix LGD, only K simulation
    % Montecarlo
    [~, EL.HP_doubleT.SG.K(dof-1,:), RC.HP_doubleT.SG.K(dof-1,:), RC_CI.HP_doubleT.SG.K(dof-1,:,:), EL_CI.HP_doubleT.SG.K(dof-1,:,:)] = montecarlo_RC_HP_T_double(N1,k_hat_SG,sigma_SG,LGD_hat,sigma_LGD,alpha,t1_T,t2_T,M1,epsilon,2,N_obligors(1));
    [~, EL.HP_doubleT.AR.K(dof-1,:), RC.HP_doubleT.AR.K(dof-1,:), RC_CI.HP_doubleT.AR.K(dof-1,:,:), EL_CI.HP_doubleT.AR.K(dof-1,:,:)] = montecarlo_RC_HP_T_double(N1,k_hat_AR,sigma_AR,LGD_hat,sigma_LGD,alpha,t1_T,t2_T,M1,epsilon,2,N_obligors(1));
    % Add-on
    [AddOn.HP_doubleT.SG.K(dof-1,:), AddOn_CI.HP_doubleT.SG.K(dof-1,:,:)] = Add_on(RC.HP_doubleT.SG.K(dof-1,:),RC.naive.SG,EL.HP_doubleT.SG.K(dof-1,:),EL.naive.SG,squeeze(RC_CI.HP_doubleT.SG.K(dof-1,:,:)), squeeze(EL_CI.HP_doubleT.SG.K(dof-1,:,:)));
    [AddOn.HP_doubleT.AR.K(dof-1,:), AddOn_CI.HP_doubleT.AR.K(dof-1,:,:)] = Add_on(RC.HP_doubleT.AR.K(dof-1,:),RC.naive.AR,EL.HP_doubleT.AR.K(dof-1,:),EL.naive.AR,squeeze(RC_CI.HP_doubleT.AR.K(dof-1,:,:)), squeeze(EL_CI.HP_doubleT.AR.K(dof-1,:,:)));
    
    % Independent case
    % Montecarlo
    [~, EL.HP_doubleT.SG.ind(dof-1,:), RC.HP_doubleT.SG.ind(dof-1,:), RC_CI.HP_doubleT.SG.ind(dof-1,:,:), EL_CI.HP_doubleT.SG.ind(dof-1,:,:)] = montecarlo_RC_HP_T_double(N1,k_hat_SG,sigma_SG,LGD_hat,sigma_LGD,alpha,t1_T,t2_T,M1,epsilon,3,N_obligors(1));
    [~, EL.HP_doubleT.AR.ind(dof-1,:), RC.HP_doubleT.AR.ind(dof-1,:), RC_CI.HP_doubleT.AR.ind(dof-1,:,:), EL_CI.HP_doubleT.AR.ind(dof-1,:,:)] = montecarlo_RC_HP_T_double(N1,k_hat_AR,sigma_AR,LGD_hat,sigma_LGD,alpha,t1_T,t2_T,M1,epsilon,3,N_obligors(1));
    % Add-on
    [AddOn.HP_doubleT.SG.ind(dof-1,:), AddOn_CI.HP_doubleT.SG.ind(dof-1,:,:)] = Add_on(RC.HP_doubleT.SG.ind(dof-1,:),RC.naive.SG,EL.HP_doubleT.SG.ind(dof-1,:),EL.naive.SG,squeeze(RC_CI.HP_doubleT.SG.ind(dof-1,:,:)), squeeze(EL_CI.HP_doubleT.SG.ind(dof-1,:,:)));
    [AddOn.HP_doubleT.AR.ind(dof-1,:), AddOn_CI.HP_doubleT.AR.ind(dof-1,:,:)] = Add_on(RC.HP_doubleT.AR.ind(dof-1,:),RC.naive.AR,EL.HP_doubleT.AR.ind(dof-1,:),EL.naive.AR,squeeze(RC_CI.HP_doubleT.AR.ind(dof-1,:,:)), squeeze(EL_CI.HP_doubleT.AR.ind(dof-1,:,:)));
    
    % Dependent case
    % Montecarlo
    [~, EL.HP_doubleT.SG.dep(dof-1,:), RC.HP_doubleT.SG.dep(dof-1,:), RC_CI.HP_doubleT.SG.dep(dof-1,:,:), EL_CI.HP_doubleT.SG.dep(dof-1,:,:)] = montecarlo_RC_HP_T_double(N1,k_hat_SG,sigma_SG,LGD_hat,sigma_LGD,alpha,t1_T,t2_T,M1,epsilon,4,N_obligors(1),corr_SG(1,2));
    [~, EL.HP_doubleT.AR.dep(dof-1,:), RC.HP_doubleT.AR.dep(dof-1,:), RC_CI.HP_doubleT.AR.dep(dof-1,:,:), EL_CI.HP_doubleT.AR.dep(dof-1,:,:)] = montecarlo_RC_HP_T_double(N1,k_hat_AR,sigma_AR,LGD_hat,sigma_LGD,alpha,t1_T,t2_T,M1,epsilon,4,N_obligors(1),corr_AR(1,2));
    % Add-on
    [AddOn.HP_doubleT.SG.dep(dof-1,:), AddOn_CI.HP_doubleT.SG.dep(dof-1,:,:)] = Add_on(RC.HP_doubleT.SG.dep(dof-1,:),RC.naive.SG,EL.HP_doubleT.SG.dep(dof-1,:),EL.naive.SG,squeeze(RC_CI.HP_doubleT.SG.dep(dof-1,:,:)), squeeze(EL_CI.HP_doubleT.SG.dep(dof-1,:,:)));
    [AddOn.HP_doubleT.AR.dep(dof-1,:), AddOn_CI.HP_doubleT.AR.dep(dof-1,:,:)] = Add_on(RC.HP_doubleT.AR.dep(dof-1,:),RC.naive.AR,EL.HP_doubleT.AR.dep(dof-1,:),EL.naive.AR,squeeze(RC_CI.HP_doubleT.AR.dep(dof-1,:,:)), squeeze(EL_CI.HP_doubleT.AR.dep(dof-1,:,:)));
    
    for i=1:length(alpha)
        fprintf("\n")
        fprintf("Add-on (alpha = %.3f - N obligors = %d - dof = %d)\n\n",alpha(i),N_obligors(1),dof)
        fprintf("Only LGD simulation:                <strong>SG</strong> = %.4f %% - CI = [%.4f, %.4f] %%   <strong>AR</strong> = %.4f %% - CI = [%.4f, %.4f] %% \n",AddOn.HP_doubleT.SG.LGD(dof-1,i),AddOn_CI.HP_doubleT.SG.LGD(dof-1,i,1),AddOn_CI.HP_doubleT.SG.LGD(dof-1,i,2),AddOn.HP_doubleT.AR.LGD(dof-1,i),AddOn_CI.HP_doubleT.AR.LGD(dof-1,i,1),AddOn_CI.HP_doubleT.AR.LGD(dof-1,i,2))
        fprintf("Only K simulation:                  <strong>SG</strong> = %.4f %% - CI = [%.4f, %.4f] %%   <strong>AR</strong> = %.4f %% - CI = [%.4f, %.4f] %% \n",AddOn.HP_doubleT.SG.K(dof-1,i),AddOn_CI.HP_doubleT.SG.K(dof-1,i,1),AddOn_CI.HP_doubleT.SG.K(dof-1,i,2),AddOn.HP_doubleT.AR.K(dof-1,i),AddOn_CI.HP_doubleT.AR.K(dof-1,i,1),AddOn_CI.HP_doubleT.AR.K(dof-1,i,2))
        fprintf("LGD - K independent simulation:     <strong>SG</strong> = %.4f %% - CI = [%.4f, %.4f] %%   <strong>AR</strong> = %.4f %% - CI = [%.4f, %.4f] %% \n",AddOn.HP_doubleT.SG.ind(dof-1,i),AddOn_CI.HP_doubleT.SG.ind(dof-1,i,1),AddOn_CI.HP_doubleT.SG.ind(dof-1,i,2),AddOn.HP_doubleT.AR.ind(dof-1,i),AddOn_CI.HP_doubleT.AR.ind(dof-1,i,1),AddOn_CI.HP_doubleT.AR.ind(dof-1,i,2))
        fprintf("LGD - K dependent simulation:       <strong>SG</strong> = %.4f %% - CI = [%.4f, %.4f] %%   <strong>AR</strong> = %.4f %% - CI = [%.4f, %.4f] %% \n\n",AddOn.HP_doubleT.SG.dep(dof-1,i),AddOn_CI.HP_doubleT.SG.dep(dof-1,i,1),AddOn_CI.HP_doubleT.SG.dep(dof-1,i,2),AddOn.HP_doubleT.AR.dep(dof-1,i),AddOn_CI.HP_doubleT.AR.dep(dof-1,i,1),AddOn_CI.HP_doubleT.AR.dep(dof-1,i,2))
    end
    fprintf("Montecarlo simulations computation time (N obligors = %d - dof = %d): ",N_obligors(1),dof)
    toc
end

%% Plot double t-student convergence

for i = 1:length(alpha)
    figure('Name','HP_doubleT convergence plot')
    
    subplot(4,2,1)
    plot(x,RC.HP_doubleT.AR.LGD(:,i),'LineWidth',2)
    hold on
    plot(x,RC.HP.AR.LGD(1,i)*ones(length(x),1),'LineWidth',2)
    hold off
    title('Simulating LGD only - AR case')
    xlabel('degree of freedom')
    ylabel('RC')
    
    subplot(4,2,2)
    plot(x,RC.HP_doubleT.SG.LGD(:,i),'LineWidth',2)
    hold on
    plot(x,RC.HP.SG.LGD(1,i)*ones(length(x),1),'LineWidth',2)
    hold off
    title('Simulating LGD only - SG case')
    xlabel('degree of freedom')
    ylabel('RC')
    
    subplot(4,2,3)
    plot(x,RC.HP_doubleT.AR.K(:,i),'LineWidth',2)
    hold on
    plot(x,RC.HP.AR.K(1,i)*ones(length(x),1),'LineWidth',2)
    hold off
    title('Simulating K only - AR case')
    xlabel('degree of freedom')
    ylabel('RC')
    
    subplot(4,2,4)
    plot(x,RC.HP_doubleT.SG.K(:,i),'LineWidth',2)
    hold on
    plot(x,RC.HP.SG.K(1,i)*ones(length(x),1),'LineWidth',2)
    hold off
    title('Simulating K only - SG case')
    xlabel('degree of freedom')
    ylabel('RC')
    
    subplot(4,2,5)
    plot(x,RC.HP_doubleT.AR.ind(:,i),'LineWidth',2)
    hold on
    plot(x,RC.HP.AR.ind(1,i)*ones(length(x),1),'LineWidth',2)
    hold off
    title('Simulating LGD and K independent - AR case')
    xlabel('degree of freedom')
    ylabel('RC')
    
    subplot(4,2,6)
    plot(x,RC.HP_doubleT.SG.ind(:,i),'LineWidth',2)
    hold on
    plot(x,RC.HP.SG.ind(1,i)*ones(length(x),1),'LineWidth',2)
    hold off
    title('Simulating LGD and K independent - SG case')
    xlabel('degree of freedom')
    ylabel('RC')
    
    subplot(4,2,7)
    plot(x,RC.HP_doubleT.SG.dep(:,i),'LineWidth',2)
    hold on
    plot(x,RC.HP.SG.dep(1,i)*ones(length(x),1),'LineWidth',2)
    hold off
    title('Simulating LGD and K dependent - AR case')
    xlabel('degree of freedom')
    ylabel('RC')
    
    subplot(4,2,8)
    plot(x,RC.HP_doubleT.SG.dep(:,i),'LineWidth',2)
    hold on
    plot(x,RC.HP.SG.dep(1,i)*ones(length(x),1),'LineWidth',2)
    hold off
    title('Simulating LGD and K dependent - SG case')
    xlabel('degree of freedom')
    ylabel('RC')

    txt = ['HP doubleT convergence plot, alpha=',num2str(alpha(i))];
    sgtitle(txt)
end

%% Standard Model

fprintf('\n\n --------------- Standard RC ---------------\n');

% Risk weights
RW_SG = 1.5; % Risk weight associated to B Corporate bonds
RW_AR = 1; % Risk weight associated to BBB Corporate bonds

% Standard Regulatory Capital
RC.Standard.SG = 0.08*RW_SG;
RC.Standard.AR = 0.08*RW_AR;

fprintf("Standard Regulatory Capitals:         <strong>SG</strong> = %.4f      <strong>AR</strong> = %.4f\n",RC.Standard.SG,RC.Standard.AR)