function [weights_0_5y weights_5_15y] = weights_vega(deltas)

% The function computes the weights in order to do the coarse-granied 
% deltas        : deltas given by the bootstrap
    
    weights_0_5y = [ones(1,length(find(deltas<5))), interp1([5,15],[1,0],round(deltas(find(deltas>=5 & deltas<15))),"linear"),...
                   zeros(1,length(find(deltas>15)))]';
    weights_5_15y = [zeros(1,length(find(deltas<5))), interp1([5,15],[0,1],round(deltas(find(deltas>=5 & deltas<=15.1))),"linear"),...
                   zeros(1,length(find(deltas>15))-1)]';     
end