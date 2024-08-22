function [spot_vol, spot_vol_mkt] = calibration_vol_1(Maturities,Volatilities,Strikes,discounts,delta,dt,price,dates)
    
    % This function computes the spot volatilities by a bootstrap using the
    % bachelier model starting from the given flat volatilities of the
    % market
    %
    % Maturities:       Cap maturities
    % Volatilities:     matrix of flat volalitilities from the table (market)
    % Strikes:          vector of strikes 
    % discounts:        discounts taken from the bootstrap ( every 3 months)
    % delta:            time intervals referred always to t0
    % dt:               time intervals from t_i to t_(i+1)
    % price:            matrix of prices of the cap computed from table 
    %                   of flat volalitilities
    % dates :           Caplets dates

    % number of caplets in a year
    num_caplets = 4; 

    % greatest maturity of the given volatilities 
    m = Maturities(end);    
    % number of spot vol: 4 for every year except the first quarter (just 3)
    M = m*num_caplets-1;
    
    % number of strikes
    n = length(Strikes);
    
    % spot vol for each quarter
    spot_vol = zeros(M,n);
    % spot vol for each maturity
    spot_vol_mkt = zeros(length(Maturities),n);

    % spot volatilities for the first cap are the same as the flat ones
    spot_vol(1,:) = Volatilities(1,:);
    spot_vol(2,:) = Volatilities(1,:);
    spot_vol(3,:) = Volatilities(1,:);

    spot_vol_mkt(1,:) = Volatilities(1,:);
    
    % Computing forward discounts and forward rates 
    forward_disc = [discounts(1), discounts(2:end)./discounts(1:end-1)];
    % Computing Libor rate
    Libor = (1./forward_disc-1)./dt;

    for i=2:length(Maturities) % rows
        for j=1:n % columns
            % last value available of spot vol
            spot_vol_scalar=spot_vol(Maturities(i-1)*num_caplets-1,j);

            % Difference of maturities betweeen one cap and the prevoius one in years
            difference_mat = Maturities(i)-Maturities(i-1);
            
            % I need to create the vector of dt that are inside the fwd cap
            % from Maturities(i-1) up to maturities(i)
            dt_cap = dt((Maturities(i-1)*num_caplets+1):Maturities(i)*num_caplets);

            % I need to create the vector of discounts that are inside the fwd cap
            discounts_cap = discounts((Maturities(i-1)*num_caplets+1):Maturities(i)*num_caplets);

            % I need to create the vector of delta that are inside the fwd cap
            delta_cap = delta((Maturities(i-1)*num_caplets):(Maturities(i)*num_caplets-1));

            % I need to create the vector of delta that are inside the fwd cap
            Libor_cap = Libor((Maturities(i-1)*num_caplets+1):Maturities(i)*num_caplets);

            % Finding the value of the forward start cap by sum of caplets with
            % spot vol as unknow x.
            % we recall the function computing_sigma which contains as a
            % vector the linear equation for the sigmas all dependent on
            % the last sigma which is x.

            f=@(x) price(i,j)-(price(i-1,j)+(sum(Pricing_Caplet(dt_cap,...
                   discounts_cap,Libor_cap,Strikes(j),computing_sigma_1(Maturities,...
                   spot_vol_scalar,i,x,dates)',delta_cap))));
            
            % using fzero we find the value for last spot sigma of our cap
            spot_vol_final = fzero(f,0.022);
            % filling the matrix using the function computing_sigma 
            % It's a just a linear interpolation
            spot_vol((Maturities(i-1)*num_caplets):(Maturities(i)*num_caplets-1),j) = computing_sigma_1(Maturities,...
                                                                    spot_vol_scalar,i,spot_vol_final,dates)';
            % filling the matrix (with shape of the market) with just the
            % sigma final
            spot_vol_mkt(i,j) = spot_vol_final;
        end
    end
end