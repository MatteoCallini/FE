function bucket = vega_bucket_spline(dates_caplets,disc_caplets,Volatilities,Maturities,Strikes,ATM_strike,Cap_Maturity,Cap_price,fwd_libor)
    
    % This function computes the vega bucket sensitivity for a structured
    % product, starting from the volatility matrix
    %
    % dates_caplets     caplets dates
    % disc_caplets      discounts caplets
    % Volatilities      Volatility matrix
    % Maturities        mkt cap maturities
    % Strikes           mkt cap strikes
    % ATM_strike        given strike
    % Cap_Maturity      cap maturity
    % Cap_price         cap initial price
    % fwd_libor         fwd libor rates
    
    actual360 = 2;
    actual365 = 3;
    sizes = size(Volatilities);
    
    % bucket contains the sensitivities
    bucket = zeros(sizes(1),1);
    d_basis_point=1e-4;

    % if we need to compute it with spot volatilities  
    delta_caplets=yearfrac(dates_caplets(1),dates_caplets(2:end),actual365);
    % discounts_caplets = exp(-zeroRates_caplets.*delta_caplets);
    dt_caplets=yearfrac(dates_caplets(1:end-1),dates_caplets(2:end),actual360);

    for i = 1:sizes(1)

        Volatilities(i,:) = Volatilities(i,:)+d_basis_point;
        % computing the price with Bachelier Formula
        price = bachelier_price(Volatilities,Strikes,disc_caplets,Maturities,delta_caplets,dt_caplets);
        % Calibrating the spot volatilities on the flat volatilities 
        [spot_vol, ~] = calibration_vol_1(Maturities,Volatilities,Strikes, ...
                                                   disc_caplets,delta_caplets,dt_caplets,price,dates_caplets);

        sigma_ATM_spot = zeros(length(spot_vol(:,1)),1);
        for t=1:length(spot_vol(:,1))
             sigma_ATM_spot(t) = interp1(Strikes, spot_vol(t,:), ATM_strike, 'spline');
        end

        bucket(i) = Cap_pricing(sigma_ATM_spot(1:Cap_Maturity*4-1,1),ATM_strike,disc_caplets(1:Cap_Maturity*4),fwd_libor(1:Cap_Maturity*4),...
                           delta_caplets(1:(Cap_Maturity*4-1)),dt_caplets(1:Cap_Maturity*4)) ...
                           - Cap_price;

        Volatilities(i,:) = Volatilities(i,:)-d_basis_point;
    end
end