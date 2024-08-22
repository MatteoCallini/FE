function [EL_vec, EL, RC, RC_CI, EL_CI] = montecarlo_RC(N,k_hat,sigma_k,LGD_cap,sigma_LGD,alpha,g,M,flag,correlation)

    % This function computes the Regulatory capital of a LHP using Vasicek 
    % model via Montecarlo approach
    % 
    % INPUTS
    % N :           Number of simulations
    % k_hat :       Mean of k (gaussian cumulative inverse default probability) values
    % sigma_k :     k values standard deviation
    % LGD_cap :     Loss Given Default mean
    % sigma_LGD :   Loss Given Default standard deviation
    % alpha :       confidence level
    % g :           std gaussian N-dim vector
    % flag :        1 -> I fix k = k_hat for each k
    %               2 -> I fix LGD as LGD_cap
    %               3 -> I consider LGD and k independent
    %               4 -> I consider LGD and k dependent
    % correlation:  Correlation between LGD and k, mandatory only if flag==4
    %
    % OUTPUTS
    % EL_vec :      Loss vector which contains the loss for each
    %               simulation
    % EL :          Expected Loss
    % RC :          Regulatory Capital
    % RC_CI :       Regulatory capital Confidence interval (confidence level = alpha)
    % EL_CI :       Expected Loss Confidence interval (confidence level = alpha)
    
    alpha_ci = 0.999; % confidence level

    if flag == 1        
        LGD_MC = LGD_cap + sigma_LGD*g(1,:);
        rho = correlation_IRB(normcdf(k_hat));
        EL_vec = LGD_MC.*normcdf((k_hat-sqrt(rho).*M)./sqrt(1-rho));
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
        k_MC = k_hat + sigma_k*g(2,:);
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
        K_MC = k_hat + sigma_k*g(1,:);
        LGD_MC = LGD_cap + sigma_LGD*g(2,:);
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

    if flag == 4
        correlation_matrix = [1 correlation; correlation 1];
        A=chol(correlation_matrix,"lower");
        x=A*g;

        K_MC = k_hat + sigma_k*x(1,:);
        LGD_MC = LGD_cap + sigma_LGD*x(2,:);
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