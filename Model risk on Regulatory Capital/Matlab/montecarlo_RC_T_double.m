function [EL_vec, EL, RC, RC_CI, EL_CI] = montecarlo_RC_T_double(N,k_hat,sigma_k,LGD_cap,sigma_LGD,alpha,t1,t2,M,flag,correlation)

    % This function computes the Regulatory capital using a Montecarlo approach
    % simulating k and LGD as double t-student
    % 
    % INPUTS
    % N :           Number of simulations
    % k_hat :       Mean of k (gaussian cumulative inverse default probability) values
    % sigma_k :     k values standard deviation
    % LGD_cap :     Loss Given Default mean
    % sigma_LGD :   Loss Given Default standard deviation
    % alpha :       confidence level
    % t1 :          t-student 2 x N matrix (LGD)
    % t2 :          t-student 2 x N matrix (k)
    % flag :        1 -> I fix k = k_hat for each k
    %               2 -> I fix LGD = LGD_cap
    %               3 -> I consider LGD and k independent
    %               4 -> I consider LGD and k dependent
    % correlation:  correlation between LGD and k, mandatory only if flag==4
    %
    % OUTPUTS
    % EL_vec :      Loss vector which contaains the loss for each
    %               simulation
    % EL :          Expected Loss
    % RC :          Regulatory Capital
    % RC_CI :       Regulatory capital Confidence interval (confidence level = alpha)
    % EL_CI :       Expected Loss Confidence interval (confidence level = alpha)
    
    % Correlation between t-students in the double t-student.
    % We can choose any coefficient in (0,1) in order to have a double
    % t-student with the same mean and variance
    
    rho_v = 0.5; % correlation double t-student
    alpha_ci = 0.999; % confidence level

    if flag == 1        
        LGD_MC = LGD_cap + sigma_LGD*(sqrt(rho_v)*t1(1,:) + sqrt(1-rho_v)*t1(2,:));
        rho = correlation_IRB(normcdf(k_hat));
        EL_vec = LGD_MC.*normcdf((k_hat-sqrt(rho)*M)/sqrt(1-rho));
        EL = mean(EL_vec);
        EL_quantile = quantile(EL_vec,alpha);
        RC = EL_quantile-EL;

        % Confidence interval computation
        for i = 1:length(alpha)
            RC_CI(i,1) = EL_quantile(i) - EL + tinv((1-alpha_ci)/2,N)*sqrt(2/N*(var(EL_vec) + (alpha(i)*(1-alpha(i)))/(exp(-EL_quantile(i)^2/2)/sqrt(2*pi))^2));
            RC_CI(i,2) = EL_quantile(i) - EL - tinv((1-alpha_ci)/2,N)*sqrt(2/N*(var(EL_vec) + (alpha(i)*(1-alpha(i)))/(exp(-EL_quantile(i)^2/2)/sqrt(2*pi))^2));            
            EL_CI(i,:) = [EL + tinv((1-alpha_ci)/2,N)*std(EL_vec)/sqrt(N), EL - tinv((1-alpha_ci)/2,N)*std(EL_vec)/sqrt(N)];
        end
    end

    if flag == 2
        k_MC = k_hat + sigma_k*(sqrt(rho_v)*t2(1,:) + sqrt(1-rho_v)*t2(2,:));
        rho = correlation_IRB(normcdf(k_MC));
        EL_vec = LGD_cap*normcdf((k_MC-sqrt(rho).*M)./sqrt(1-rho));
        EL = mean(EL_vec);
        EL_quantile = quantile(EL_vec,alpha);
        RC = EL_quantile-EL;
        
        % Confidence interval computation
        for i = 1:length(alpha)
            RC_CI(i,1) = EL_quantile(i) - EL + tinv((1-alpha_ci)/2,N)*sqrt(2/N*(var(EL_vec) + (alpha(i)*(1-alpha(i)))/(exp(-EL_quantile(i)^2/2)/sqrt(2*pi))^2));
            RC_CI(i,2) = EL_quantile(i) - EL - tinv((1-alpha_ci)/2,N)*sqrt(2/N*(var(EL_vec) + (alpha(i)*(1-alpha(i)))/(exp(-EL_quantile(i)^2/2)/sqrt(2*pi))^2));            
            EL_CI(i,:) = [EL + tinv((1-alpha_ci)/2,N)*std(EL_vec)/sqrt(N), EL - tinv((1-alpha_ci)/2,N)*std(EL_vec)/sqrt(N)];
        end
    end
    
    if flag == 3
        LGD_MC = LGD_cap + sigma_LGD*(sqrt(rho_v)*t1(1,:) + sqrt(1-rho_v)*t1(2,:));
        k_MC = k_hat + sigma_k*(sqrt(rho_v)*t2(1,:) + sqrt(1-rho_v)*t2(2,:));
        rho = correlation_IRB(normcdf(k_MC));
        EL_vec = LGD_MC.*normcdf((k_MC-sqrt(rho).*M)./sqrt(1-rho));
        EL = mean(EL_vec);
        EL_quantile = quantile(EL_vec,alpha);
        RC = EL_quantile-EL;

        % Confidence interval computation
        for i = 1:length(alpha)
            RC_CI(i,1) = EL_quantile(i) - EL + tinv((1-alpha_ci)/2,N)*sqrt(2/N*(var(EL_vec) + (alpha(i)*(1-alpha(i)))/(exp(-EL_quantile(i)^2/2)/sqrt(2*pi))^2));
            RC_CI(i,2) = EL_quantile(i) - EL - tinv((1-alpha_ci)/2,N)*sqrt(2/N*(var(EL_vec) + (alpha(i)*(1-alpha(i)))/(exp(-EL_quantile(i)^2/2)/sqrt(2*pi))^2));            
            EL_CI(i,:) = [EL + tinv((1-alpha_ci)/2,N)*std(EL_vec)/sqrt(N), EL - tinv((1-alpha_ci)/2,N)*std(EL_vec)/sqrt(N)];
        end
    end

    if flag == 4
        sigma=[1 correlation; correlation 1];
        A=chol(sigma,"lower");
        x=A*[(sqrt(rho_v)*t1(1,:) + sqrt(1-rho_v)*t1(2,:)); (sqrt(rho_v)*t2(1,:) + sqrt(1-rho_v)*t2(2,:))];
        K_MC = k_hat + sigma_k*x(2,:);
        LGD_MC = LGD_cap + sigma_LGD*x(1,:);
        rho = correlation_IRB(normcdf(K_MC));
        EL_vec = LGD_MC.*normcdf((K_MC-sqrt(rho).*M)./sqrt(1-rho));
        EL = mean(EL_vec);
        EL_quantile = quantile(EL_vec,alpha);
        RC = EL_quantile-EL;

        % Confidence interval computation
        for i = 1:length(alpha)
            RC_CI(i,1) = EL_quantile(i) - EL + tinv((1-alpha_ci)/2,N)*sqrt(2/N*(var(EL_vec) + (alpha(i)*(1-alpha(i)))/(exp(-EL_quantile(i)^2/2)/sqrt(2*pi))^2));
            RC_CI(i,2) = EL_quantile(i) - EL - tinv((1-alpha_ci)/2,N)*sqrt(2/N*(var(EL_vec) + (alpha(i)*(1-alpha(i)))/(exp(-EL_quantile(i)^2/2)/sqrt(2*pi))^2));            
            EL_CI(i,:) = [EL + tinv((1-alpha_ci)/2,N)*std(EL_vec)/sqrt(N), EL - tinv((1-alpha_ci)/2,N)*std(EL_vec)/sqrt(N)];
        end
    end
end