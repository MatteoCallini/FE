function sigma = computing_sigma_1(Maturities,spot_vol_scalar,i,spot_final,dates)

    % This function built the right number of equation (linear equation) 
    % in order to obtain the spot vols
    % Maturities        : vector containg the maturities of the caps
    % Spot_vol_scalar   : value of the spot vol of the previous cap
    % i                 : index of the row of the given matrix of
    %                     volatilities we are dealing with
    % delta             : time intervals referred always to t0
    % difference_mat    : difference of maturities betweeen one cap 
    %                     and the prevoius one in years
    % spot_final        : value of the spot vol corresponding to the final
    %                     caplet of our cap

    % number of caplets in a year
    num_caplets = 4; 
    % defining a vector for the spot vol
    sigma = interp1([dates(Maturities(i-1)*num_caplets-2),dates(Maturities(i)*num_caplets-2)],...
                    [spot_vol_scalar,spot_final],...
                    dates((Maturities(i-1)*num_caplets-1):(Maturities(i)*num_caplets-2)))';
end