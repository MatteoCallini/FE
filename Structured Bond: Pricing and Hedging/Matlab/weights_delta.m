function [weights_0_2y weights_2_5y weights_5_10y weights_10_15y] = weights_delta(deltas)
% The function computes the weights in order to do the coarse-granied 
% deltas        : deltas given by the bootstrap

    weights_0_2y = [ones(1,length(find(deltas<2))), interp1([2,5],[1,0],round(deltas(find(deltas>=2 & deltas<5))),"linear"),...
                   zeros(1,length(find(deltas>=5)))]';
    weights_2_5y = [zeros(1,length(find(deltas<2))), interp1([2,5],[0,1],round(deltas(find(deltas>=2 & deltas<5))),"linear"),...
                   interp1([5,10],[1,0],round(deltas(find(deltas>=5 & deltas<10))),"linear"),...
                   zeros(1,length(find(deltas>=10)))]';
    weights_5_10y = [zeros(1,length(find(deltas<5))), interp1([5,10],[0,1],round(deltas(find(deltas>=5 & deltas<10))),"linear"),...
                   interp1([10,15],[1,0],round(deltas(find(deltas>=10 & deltas<15))),"linear"),...
                   zeros(1,length(find(deltas>=15)))]';
    weights_10_15y = [zeros(1,length(find(deltas<10))), interp1([10,15],[0,1],round(deltas(find(deltas>=10 & deltas<15))),"linear"),...
                   interp1([15,20],[1,0],round(deltas(find(deltas>=15,1,'first'))),"linear"),...
                   zeros(1,length(find(deltas>=15))-1)]'; 
end