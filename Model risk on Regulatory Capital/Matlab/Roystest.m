function [H, P] = Roystest(X,alpha)
% ROYSTEST. Royston's Multivariate Normality Test. 
% It is well known that many multivariate statistical procedures call upon
% the assumption of multivariate normality (MVN). At least 50 different
% procedures have been proposed for this problem. Unfortunately, different
% conclusions about the MVN of a data set can be reached by different
% procedures (Mecklin and Mundfrom, 2005).
%
% The Shapiro-Wilk test (Shapiro and Wilk, 1965), is generally considered 
% to be an excellent test of univariate normality. It is only natural to 
% extend it to the multivariate case, as done by Royston (1982). Simulation
% results have show Royston’s test to have very good Type I error control
% and power against many different alternative distributions. Further, 
% Royston’s test involves a rather ingenious correction for the correlation
% between the variables in the sample.
%
% Royston’s (1983) marginal method first tests each of the p variates for
% univariate normality with a Shapiro-Wilk statistic, then combines the p
% dependent tests into one omnibus test statistic for multivariate
% normality. Royston transforms the p-Shapiro-Wilk statistics into what he
% claims is an approximately Chi-squared random variable, with e (1< e <p)
% degrees of freedom. The degrees of freedom are estimated by taking into
% account possible correlation structures between the original p test
% statistics. This test has been found to behave well when the sample size
% is small and the variates are relatively uncorrelated (Mecklin and
% Mundfrom, 2005).
%
% Let W_j denote the value of the Shapiro-Wilk statistic for the jth
% variable in a p-variate distribution. Then,
%
%            R_j = { pHi^-1 [1/2Phi{-((1 - W_j)^g - m)/s}] }^2 ,
%               
% where g, m,and s are calculated from polynomial approximations and Phi^-1
% and Phi(.) denotes, respectivelly, the inverse and standard normal cdf.
% If the data are MVN,
%                               _p _  
%                               \             
%                       H  =  e /_ _ R_j/p
%                               j=1   
%
% is approximately Chi-square distributed, where the equivalent degrees of
% freedom is,
%                     e = p/[1 + (p - 1) mC],
%
% where mC is an estimate of the average correlation among the R_j's. This
% Chi-square distribution is used to obtain the critical or P-value for the
% MVN test.
%
% This m-file has a function to generate the Shapiro-Wilk's W statistic 
% needed to feed the Royston's test for multivariate normality. It was
% taken from the Fortran Algorithm AS R94 (Royston, 1995) [URL address
% http://lib.stat.cmu.edu/apstat/181]. Here, we present both the options
% for the sample kurtosis type: 1) Shapiro-Francia for leptokurtic, and 
% 2) Shapiro-Wilk for the platykurtic ones. 
%
% Syntax: function Roystest(X,alpha) 
%      
% Inputs:
%      X - data matrix (Size of matrix must be n-by-p; data=rows,
%          indepent variable=columns) 
%  alpha - significance level (default = 0.05)
%
% Output:
%        - Royston's Multivariate Normality Test
%
% Example: From the Table 11.5 (Iris data) of Johnson and Wichern (1992,
%          p. 562), we take onlt the Isis setosa data to test if it has a 
%          multivariate normality distribution using the Doornik-Hansen
%          omnibus test. Data has 50 observations on 4 independent 
%          variables (Var1 (x1) = sepal length; Var2 (x2) = sepal width; 
%          Var3 (x3) = petal length; Var4 (x4) = petal width. 
%
%                      ------------------------------
%                        x1      x2      x3      x4       
%                      ------------------------------
%                       5.1     3.5     1.4     0.2     
%                       4.9     3.0     1.4     0.2     
%                       4.7     3.2     1.3     0.2     
%                       4.6     3.1     1.5     0.2     
%                       5.0     3.6     1.4     0.2     
%                       5.4     3.9     1.7     0.4     
%                        .       .       .       .      
%                        .       .       .       .      
%                        .       .       .       .      
%                       5.1     3.8     1.6     0.2     
%                       4.6     3.2     1.4     0.2     
%                       5.3     3.7     1.5     0.2     
%                       5.0     3.3     1.4     0.2     
%                      ------------------------------
%
% Total data matrix must be:
% You can get the X-matrix by calling to iris data file provided in
% the zip as 
%          load path-drive:iriset 
% 
% Calling on Matlab the function: 
%          Roystest(X)
%
% Answer is:
%
% Royston's Multivariate Normality Test
% -------------------------------------------------------------------
% Number of variables: 4
% Sample size: 50
% -------------------------------------------------------------------
% Royston's statistic: 31.518027
% Equivalent degrees of freedom: 3.923160
% P-value associated to the Royston's statistic: 0.000002
% With a given significance = 0.050
% Data analyzed do not have a normal distribution.
% -------------------------------------------------------------------
%
% Created by A. Trujillo-Ortiz, R. Hernandez-Walls, K. Barba-Rojo, 
%             and L. Cupul-Magana
%             Facultad de Ciencias Marinas
%             Universidad Autonoma de Baja California
%             Apdo. Postal 453
%             Ensenada, Baja California
%             Mexico.
%             atrujo@uabc.mx
%
% Copyright. November 20, 2007.
%
% To cite this file, this would be an appropriate format:
% Trujillo-Ortiz, A., R. Hernandez-Walls, K. Barba-Rojo and
%   L. Cupul-Magana. (2007). Roystest:Royston's Multivariate Normality Test.   
%   A MATLAB file. [WWW document]. URL http://www.mathworks.com/
%   matlabcentral/fileexchange/loadFile.do?objectId=17811
%
% References:
% Johnson, R.A. and Wichern, D. W. (1992). Applied Multivariate Statistical
%      Analysis. 3rd. ed. New-Jersey:Prentice Hall.
% Mecklin, C.J. and Mundfrom, D.J. (2005). A Monte Carlo comparison of the
%      Type I and Type II error rates of tests of multivariate normality.
%      Journal of Statistical Computation and Simulation, 75:93-107.
% Royston, J.P. (1982). An Extension of Shapiro and Wilk’s W Test for
%      Normality to Large Samples. Applied Statistics, 31(2):115–124.
% Royston, J.P. (1983). Some Techniques for Assessing Multivariate
%      Normality Based on the Shapiro-Wilk W. Applied Statistics, 32(2):
% Royston, J.P. (1992). Approximating the Shapiro-Wilk W-Test for non-
%      normality. Statistics and Computing, 2:117-119.
%      121–133.
% Royston, J.P. (1995). Remark AS R94: A remark on Algorithm AS 181: The W
%      test for normality. Applied Statistics, 44:547-551.
% Shapiro, S. and Wilk, M. (1965). An analysis of variance test for 
%      normality. Biometrika, 52:591–611.
%
Roystest = 1;
if nargin < 2,
    alpha = 0.05; %default
end

if (alpha <= 0 || alpha >= 1),
    fprintf(['Warning: Significance level error; must be 0 <'...
        ' alpha < 1 \n']);
    return;
end

if nargin < 1,
    error('Requires at least one input argument.');
    return;
end

[n,p] = size(X);

if (n <= 3),
    error('n is too small.');
    return,
elseif (n >= 4) && (n <=11),
    x = n;
    g = -2.273 + 0.459*x;
    m = 0.5440 - 0.39978*x + 0.025054*x^2 - 0.0006714*x^3;
    s = exp(1.3822 - 0.77857*x + 0.062767*x^2 - 0.0020322*x^3); 
    for j = 1:p,
        W(j) = ShaWilstat(X(:,j));
        Z(j) = (-log(g - (log(1 - W(j)))) - m)/s;
    end
elseif (n >= 12) && (n <=2000),
    x = log(n);
    g = 0;
    m = -1.5861 - 0.31082*x - 0.083751*x^2 + 0.0038915*x^3;
    s = exp(-0.4803 -0.082676*x + 0.0030302*x^2);  
    for j = 1:p,
        W(j) = ShaWilstat(X(:,j));
        Z(j) = ((log(1 - W(j))) + g - m)/s;
    end
else
    error('n is not in the proper size range.'); %error('n is too large.');return,
    return,
end

for j = 1:p,
    R(j) = (norminv((normcdf( - Z(j)))/2))^2;
end

u = 0.715;
v = 0.21364 + 0.015124*(log(n))^2 - 0.0018034*(log(n))^3;
l = 5;
C = corrcoef(X); %correlation matrix
NC = (C.^l).*(1 - (u*(1 - C).^u)/v); %transformed correlation matrix
T = sum(sum(NC)) - p; %total
mC = T/(p^2 - p); %average correlation
e = p/(1 + (p - 1)*mC); %equivalent degrees of freedom
H = (e*(sum(R)))/p; %Royston's statistic
P = 1 - chi2cdf(H,e); %P-value

disp(' ')
disp('Royston''s Multivariate Normality Test')
disp('-------------------------------------------------------------------')
fprintf('Number of variables: %i\n', p);
fprintf('Sample size: %i\n', n);
disp('-------------------------------------------------------------------')
fprintf('Royston''s statistic: %3.6f\n', H);
fprintf('Equivalent degrees of freedom: %3.6f\n', e);
fprintf('P-value associated to the Royston''s statistic: %3.6f\n', P);
fprintf('With a given significance = %3.3f\n', alpha);
if P >= alpha;
    disp('Data analyzed have a normal distribution.');
else
    disp('Data analyzed do not have a normal distribution.');
end
disp('-------------------------------------------------------------------')

% %-----------------------------------
% function [W] = ShaWilstat(x)
% %SHAWILTEST Shapiro-Wilk' W statistic for assessing a sample normality.
% % This m-file is generating from the Fortran Algorithm AS R94 (Royston,
% % 1995) [URL address http://lib.stat.cmu.edu/apstat/181]. Here we take only
% % the procedure to generate the Shapiro-Wilk's W statistic, needed to feed
% % the Royston's test for multivariate normality. Here, we present both the
% % options for the sample kurtosis type: 1) Shapiro-Francia for leptokurtic,
% % and 2) Shapiro-Wilk for the platykurtic ones. Do not assume that the
% % result of the Shapiro-Wilk test is clear evidence of normality or non-
% % normality, it is just one piece of evidence that can be helpful.
% %
% % Input:
% %      x - data vector (3 < n < 5000)
% %
% % Output:
% %      W - Shapiro-Wilk's W statistic
% %
% % Example: From the example given by Scholtz and Stephens (1987, p.922). We
% % only take the data set from laboratory 1 of eight measurements of the
% % smoothness of a certain type of paper:38.7,41.5,43.8,44.5,45.5,46.0,47.7,
% % 58.0
% %
% % Data vector is:
% %  x=[38.7,41.5,43.8,44.5,45.5,46.0,47.7,58.0];
% %
% % Calling on Matlab the function: 
% %          W = ShaWilstat(x)
% %
% % Answer is:
% %
% % W = 0.8476
% %
% % Created by A. Trujillo-Ortiz, R. Hernandez-Walls, K. Barba-Rojo, 
% %             and L. Cupul-Magana
% %             Facultad de Ciencias Marinas
% %             Universidad Autonoma de Baja California
% %             Apdo. Postal 453
% %             Ensenada, Baja California
% %             Mexico.
% %             atrujo@uabc.mx
% %
% % Copyright. November 18, 2007.
% %
% % Reference:
% % Scholz, F.W. and Stephens, M.A. (1987), K-Sample Anderson-Darling Tests.
% %     Journal of the American Statistical Association, 82:918-924.
% %
% 
% x  =  x(:); %put data in a column vector
% n = length(x); %sample size
% 
% if n < 3,
%    error('Sample vector must have at least 3 valid observations.');
% end
% 
% if n > 5000,
%     warning('Shapiro-Wilk statistic might be inaccurate due to large sample size ( > 5000).');
% end
% 
% x = sort(x); %sorting of data vector in ascending order
% m = norminv(((1:n)' - 3/8) / (n + 0.25));
% w = zeros(n,1); %preallocating weights
% 
% if kurtosis(x) > 3, %Shapiro-Francia test is better for leptokurtic samples
%     w = 1/sqrt(m'*m) * m;
%     W = (w' * x) ^2 / ((x - mean(x))' * (x - mean(x)));
% else %Shapiro-Wilk test is better for platykurtic samples
%     c = 1/sqrt(m' * m) * m;
%     u = 1/sqrt(n);
%     p1 = [-2.706056,4.434685,-2.071190,-0.147981,0.221157,c(n)];
%     p2 = [-3.582633,5.682633,-1.752461,-0.293762,0.042981,c(n-1)];
% 
%     w(n) = polyval(p1,u);
%     w(1) = -w(n);
% 
%     if n == 3,
%         w(1) = 0.707106781;
%         w(n) = -w(1);
%     end
% 
%     if n >= 6,
%         w(n-1) = polyval(p2,u);
%         w(2) = -w(n-1);
% 
%         ct =  3;
%         phi = (m'*m - 2 * m(n)^2 - 2 * m(n-1)^2) / ...
%                 (1 - 2 * w(n)^2 - 2 * w(n-1)^2);
%     else
%         ct = 2;
%         phi = (m'*m - 2 * m(n)^2) / (1 - 2 * w(n)^2);
%     end
% 
%     w(ct:n-ct+1) = m(ct:n-ct+1) / sqrt(phi);
% 
%     W = (w' * x) ^2 / ((x - mean(x))' * (x - mean(x)));
% end
% return,

function [W] = ShaWilstat(x, alpha)
%SWTEST Shapiro-Wilk parametric hypothesis test of composite normality.
%   [H, pValue, SWstatistic] = SWTEST(X, ALPHA) performs the
%   Shapiro-Wilk test to determine if the null hypothesis of
%   composite normality is a reasonable assumption regarding the
%   population distribution of a random sample X. The desired significance 
%   level, ALPHA, is an optional scalar input (default = 0.05).
%
%   The Shapiro-Wilk and Shapiro-Francia null hypothesis is: 
%   "X is normal with unspecified mean and variance."
%
%   This is an omnibus test, and is generally considered relatively
%   powerful against a variety of alternatives.
%   Shapiro-Wilk test is better than the Shapiro-Francia test for
%   Platykurtic sample. Conversely, Shapiro-Francia test is better than the
%   Shapiro-Wilk test for Leptokurtic samples.
%
%   When the series 'X' is Leptokurtic, SWTEST performs the Shapiro-Francia
%   test, else (series 'X' is Platykurtic) SWTEST performs the
%   Shapiro-Wilk test.
% 
%    [H, pValue, SWstatistic] = SWTEST(X, ALPHA)
%
% Inputs:
%   X - a vector of deviates from an unknown distribution. The observation
%     number must exceed 3 and less than 5000.
%
% Optional inputs:
%   ALPHA - The significance level for the test (default = 0.05).
%  
% Outputs:
%  SWstatistic - The test statistic (non normalized).
%
%   pValue - is the p-value, or the probability of observing the given
%     result by chance given that the null hypothesis is true. Small values
%     of pValue cast doubt on the validity of the null hypothesis.
%
%     H = 0 => Do not reject the null hypothesis at significance level ALPHA.
%     H = 1 => Reject the null hypothesis at significance level ALPHA.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Copyright (c) 17 March 2009 by Ahmed Ben Saïda          %
%                 Department of Finance, IHEC Sousse - Tunisia           %
%                       Email: ahmedbensaida@yahoo.com                   %
%                    $ Revision 3.0 $ Date: 18 Juin 2014 $               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%
% References:
%
% - Royston P. "Remark AS R94", Applied Statistics (1995), Vol. 44,
%   No. 4, pp. 547-551.
%   AS R94 -- calculates Shapiro-Wilk normality test and P-value
%   for sample sizes 3 <= n <= 5000. Handles censored or uncensored data.
%   Corrects AS 181, which was found to be inaccurate for n > 50.
%   Subroutine can be found at: http://lib.stat.cmu.edu/apstat/R94
%
% - Royston P. "A pocket-calculator algorithm for the Shapiro-Francia test
%   for non-normality: An application to medicine", Statistics in Medecine
%   (1993a), Vol. 12, pp. 181-184.
%
% - Royston P. "A Toolkit for Testing Non-Normality in Complete and
%   Censored Samples", Journal of the Royal Statistical Society Series D
%   (1993b), Vol. 42, No. 1, pp. 37-43.
%
% - Royston P. "Approximating the Shapiro-Wilk W-test for non-normality",
%   Statistics and Computing (1992), Vol. 2, pp. 117-119.
%
% - Royston P. "An Extension of Shapiro and Wilk's W Test for Normality
%   to Large Samples", Journal of the Royal Statistical Society Series C
%   (1982a), Vol. 31, No. 2, pp. 115-124.
%

%
% Ensure the sample data is a VECTOR.
%

if numel(x) == length(x)
    x  =  x(:);               % Ensure a column vector.
else
    error(' Input sample ''X'' must be a vector.');
end

%
% Remove missing observations indicated by NaN's and check sample size.
%

x  =  x(~isnan(x));

if length(x) < 3
   error(' Sample vector ''X'' must have at least 3 valid observations.');
end

if length(x) > 5000
    warning('Shapiro-Wilk test might be inaccurate due to large sample size ( > 5000).');
end

%
% Ensure the significance level, ALPHA, is a 
% scalar, and set default if necessary.
%

if (nargin >= 2) && ~isempty(alpha)
   if ~isscalar(alpha)
      error(' Significance level ''Alpha'' must be a scalar.');
   end
   if (alpha <= 0 || alpha >= 1)
      error(' Significance level ''Alpha'' must be between 0 and 1.'); 
   end
else
   alpha  =  0.05;
end

% First, calculate the a's for weights as a function of the m's
% See Royston (1992, p. 117) and Royston (1993b, p. 38) for details
% in the approximation.

x       =   sort(x); % Sort the vector X in ascending order.
n       =   length(x);
mtilde  =   norminv(((1:n)' - 3/8) / (n + 1/4));
weights =   zeros(n,1); % Preallocate the weights.

if kurtosis(x) > 3
    
    % The Shapiro-Francia test is better for leptokurtic samples.
    
    weights =   1/sqrt(mtilde'*mtilde) * mtilde;

    %
    % The Shapiro-Francia statistic W' is calculated to avoid excessive
    % rounding errors for W' close to 1 (a potential problem in very
    % large samples).
    %

    W   =   (weights' * x)^2 / ((x - mean(x))' * (x - mean(x)));

    % Royston (1993a, p. 183):
    nu      =   log(n);
    u1      =   log(nu) - nu;
    u2      =   log(nu) + 2/nu;
    mu      =   -1.2725 + (1.0521 * u1);
    sigma   =   1.0308 - (0.26758 * u2);

    newSFstatistic  =   log(1 - W);

    %
    % Compute the normalized Shapiro-Francia statistic and its p-value.
    %

    NormalSFstatistic =   (newSFstatistic - mu) / sigma;
    
    % Computes the p-value, Royston (1993a, p. 183).
    pValue   =   1 - normcdf(NormalSFstatistic, 0, 1);
    
else
    
    % The Shapiro-Wilk test is better for platykurtic samples.

    c    =   1/sqrt(mtilde'*mtilde) * mtilde;
    u    =   1/sqrt(n);

    % Royston (1992, p. 117) and Royston (1993b, p. 38):
    PolyCoef_1   =   [-2.706056 , 4.434685 , -2.071190 , -0.147981 , 0.221157 , c(n)];
    PolyCoef_2   =   [-3.582633 , 5.682633 , -1.752461 , -0.293762 , 0.042981 , c(n-1)];

    % Royston (1992, p. 118) and Royston (1993b, p. 40, Table 1)
    PolyCoef_3   =   [-0.0006714 , 0.0250540 , -0.39978 , 0.54400];
    PolyCoef_4   =   [-0.0020322 , 0.0627670 , -0.77857 , 1.38220];
    PolyCoef_5   =   [0.00389150 , -0.083751 , -0.31082 , -1.5861];
    PolyCoef_6   =   [0.00303020 , -0.082676 , -0.48030];

    PolyCoef_7   =   [0.459 , -2.273];

    weights(n)   =   polyval(PolyCoef_1 , u);
    weights(1)   =   -weights(n);
    
    if n > 5
        weights(n-1) =   polyval(PolyCoef_2 , u);
        weights(2)   =   -weights(n-1);
    
        count  =   3;
        phi    =   (mtilde'*mtilde - 2 * mtilde(n)^2 - 2 * mtilde(n-1)^2) / ...
                (1 - 2 * weights(n)^2 - 2 * weights(n-1)^2);
    else
        count  =   2;
        phi    =   (mtilde'*mtilde - 2 * mtilde(n)^2) / ...
                (1 - 2 * weights(n)^2);
    end
        
    % Special attention when n = 3 (this is a special case).
    if n == 3
        % Royston (1992, p. 117)
        weights(1)  =   1/sqrt(2);
        weights(n)  =   -weights(1);
        phi = 1;
    end

    %
    % The vector 'WEIGHTS' obtained next corresponds to the same coefficients
    % listed by Shapiro-Wilk in their original test for small samples.
    %

    weights(count : n-count+1)  =  mtilde(count : n-count+1) / sqrt(phi);

    %
    % The Shapiro-Wilk statistic W is calculated to avoid excessive rounding
    % errors for W close to 1 (a potential problem in very large samples).
    %

    W   =   (weights' * x) ^2 / ((x - mean(x))' * (x - mean(x)));

    %
    % Calculate the normalized W and its significance level (exact for
    % n = 3). Royston (1992, p. 118) and Royston (1993b, p. 40, Table 1).
    %

    newn    =   log(n);

    if (n >= 4) && (n <= 11)
    
        mu      =   polyval(PolyCoef_3 , n);
        sigma   =   exp(polyval(PolyCoef_4 , n));    
        gam     =   polyval(PolyCoef_7 , n);
    
        newSWstatistic  =   -log(gam-log(1-W));
    
    elseif n > 11
    
        mu      =   polyval(PolyCoef_5 , newn);
        sigma   =   exp(polyval(PolyCoef_6 , newn));
    
        newSWstatistic  =   log(1 - W);
    
    elseif n == 3
        mu      =   0;
        sigma   =   1;
        newSWstatistic  =   0;
    end

    %
    % Compute the normalized Shapiro-Wilk statistic and its p-value.
    %

    NormalSWstatistic   =   (newSWstatistic - mu) / sigma;
    
    % NormalSWstatistic is referred to the upper tail of N(0,1),
    % Royston (1992, p. 119).
    pValue       =   1 - normcdf(NormalSWstatistic, 0, 1);
    
    % Special attention when n = 3 (this is a special case).
    if n == 3
        pValue  =   6/pi * (asin(sqrt(W)) - asin(sqrt(3/4)));
        % Royston (1982a, p. 121)
    end
    
end

%
% To maintain consistency with existing Statistics Toolbox hypothesis
% tests, returning 'H = 0' implies that we 'Do not reject the null 
% hypothesis at the significance level of alpha' and 'H = 1' implies 
% that we 'Reject the null hypothesis at significance level of alpha.'
%

H  = (alpha >= pValue);