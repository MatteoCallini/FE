function [EL_vec, EL, RC, RC_CI, EL_CI] = montecarlo_RC_HP(N,k_hat,sigma_k,LGD_cap,sigma_LGD,alpha,g,M,epsilon,flag,N_obligors,correlation)

    % This function computes the Regulatory capital using a Montecarlo
    % approach in the Homogeneous portfolio case (finite Number of obligors)
    % 
    % INPUTS
    % N :           Number of simulations
    % k_hat :       Mean of k (gaussian cumulative inverse default probability) values
    % sigma_k :     k values standard deviation
    % LGD_cap :     Loss Given Default mean
    % sigma_LGD :   Loss Given Default standard deviation
    % alpha :       confidence level
    % g :           std gaussian N-dim vector
    % flag :        1 -> I fix k = k_hat
    %               2 -> I fix LGD = LGD_cap 
    %               3 -> I consider LGD and k independent
    %               4 -> I consider LGD and k dependent
    % N_obligors :   Number of obligors
    % correlation : Correlation matrix (mandatory for flag = 4)
    %
    % OUTPUTS
    % EL_vec :      Loss vector which contains the loss for each
    %               simulation
    % EL :          Expected Loss
    % RC :          Regulatory Capital
    % RC_CI :       Regulatory capital Confidence interval (confidence level = alpha)
    % EL_CI :       Expected Loss Confidence interval (confidence level = alpha)

    X = zeros(N_obligors,N);
    alpha_ci = 0.999; % confidence level

    if flag == 1
        k_MC = k_hat;
        LGD_MC = LGD_cap + sigma_LGD*g(2,:);
        rho = correlation_IRB(normcdf(k_MC));
        
        % Risk factors
        X = sqrt(rho).*M + sqrt(1-rho).*epsilon;

        % Defaults
        Defaults = X < k_MC;

        % Total Losses
        Loss = LGD_MC.*Defaults;

        EL_vec = mean(Loss,1);
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
        % Market factors
        k_MC = k_hat + sigma_k*g(1,:);
        LGD_MC = LGD_cap;

        rho = correlation_IRB(normcdf(k_MC));

        % Risk factors
        X = sqrt(rho).*M + sqrt(1-rho).*epsilon;

        % Defaults
        Defaults = X < k_MC;

        % Total Losses
        Loss = LGD_MC.*Defaults;
        EL_vec = mean(Loss,1);
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
        k_MC = k_hat + sigma_k*g(1,:);
        LGD_MC = LGD_cap + sigma_LGD*g(2,:);
        rho = correlation_IRB(normcdf(k_MC));
        
        % Risk factors
        X = sqrt(rho).*M + sqrt(1-rho).*epsilon;

        % Defaults
        Defaults = X < k_MC;

        % Total Losses
        Loss = LGD_MC.*Defaults;

        EL_vec = mean(Loss,1);
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
        % Market factors
        g = chol(correlation,"lower")*g;
        k_MC = k_hat + sigma_k*g(1,:);
        LGD_MC = LGD_cap + sigma_LGD*g(2,:);

        rho = correlation_IRB(normcdf(k_MC));
        
        % Risk factors
        X = sqrt(rho).*M + sqrt(1-rho).*epsilon;

        % Defaults
        Defaults = X < k_MC;

        % Total Losses
        Loss = LGD_MC.*Defaults;

        EL_vec = mean(Loss,1);
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