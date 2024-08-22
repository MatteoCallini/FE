function R = correlation_IRB(PD) 

    % This function computes the correlation between assets chosen by the
    % Basel Committee
    %
    % INPUT
    % PD :      Probability to default
    %
    % OUTPUT
    % R :       Correlation coefficient

    Rmin = 0.12;
    Rmax = 0.24;
    k = 50;
    R = Rmin*(1-exp(-k*PD))./(1-exp(-k)) + Rmax*(1-(1-exp(-k*PD))./(1-exp(-k)));  
end