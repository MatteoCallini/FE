function bucket_spot = vega_bucket(N,T,sf,s,s1,c1,c2,c3,dates_caplets,disc_caplets,Volatilities,Maturities,Strikes,X_spot,NPV_spot)
    
    % This function computes the vega bucket sensitivity for a structured
    % product, starting from the volatility matrix
    %
    % Volatilities: volatility matrix
    
    actual365 = 3;
    actual360 = 2;
    sizes = size(Volatilities);
    
    % bucket contains the sensitivities
    bucket_spot = zeros(sizes(1),1);
    d_basis_point=1e-4;

    % if we need to compute it with spot volatilities  
    delta_caplets=yearfrac(dates_caplets(1),dates_caplets(2:end),actual365);
    dt_caplets=yearfrac(dates_caplets(1:end-1),dates_caplets(2:end),actual360);

    for i = 1:sizes(1)
        Volatilities(i,:) = Volatilities(i,:)+d_basis_point;
        % computing the price with Bachelier Formula
        price = bachelier_price(Volatilities,Strikes,disc_caplets,Maturities,delta_caplets,dt_caplets);

        % Calibrating the spot volatilities on the flat volatilities 
        [spot_vol, ~] = calibration_vol_1(Maturities,Volatilities,Strikes,disc_caplets,delta_caplets,dt_caplets,price,dates_caplets);
        bucket_spot(i) = structurebond_spot(N,T,sf,s,s1,c1,c2,c3,...
                                dates_caplets,disc_caplets,spot_vol,...
                                Strikes,X_spot)-NPV_spot;
        Volatilities(i,:) = Volatilities(i,:)-d_basis_point;
    end
end