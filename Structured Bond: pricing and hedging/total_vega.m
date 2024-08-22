function totalVega_Spot_NPV = total_vega(Maturities,Volatilities,Strikes,NPV_spot,N,T,sf,s,s1,c1,c2,c3,dates_caplets,disc_caplets,X)
    
    % Computing the delta bucket for our portfolio made of the certificate
    %
    % Maturities : cap maturities
    % Volatilities : volatilities matrix
    % Strikes : volatility matrix Strikes
    % NPV_spot : NPV_spot (not exactly zero)
    % X : percentage of the principal amount computed with spot
    %     volatilities
    % N  : Notional
    % T  : Maturity
    % sf : fixed coupon (1st quarter)
    % s  : fixed part coupon of the fixed leg
    % s1 : fixed part coupon of the floating leg
    % c1 : cap for the quarters between 0.25-5 years floating coupon 
    % c2 : cap for the quarters between 5.25-10 years floating coupon 
    % c3 : cap for the quarters between 10.25-15 years floating coupon
    % dates_cplets : caplets dates
    % spot_vol : volatility matrix
    
    
    actual360 = 2;      %depo day count
    actual365 = 3;      %IB day count
    
    % bump all flat volatilities up by 1bps
    d_basis_point = 1e-4;
    Vol_shift = Volatilities+d_basis_point;
    
    % if we need to compute it with spot volatilities
    delta_caplets = yearfrac(dates_caplets(1),dates_caplets(2:end),actual365);
    dt_caplets = yearfrac(dates_caplets(1:end-1),dates_caplets(2:end),actual360);

    % computing the price with Bachelier Formula
    price = bachelier_price(Vol_shift,Strikes,disc_caplets,Maturities,delta_caplets,dt_caplets);
    
    % Calibrating the spot volatilities on the flat volatilities 
    [spot_vol, ~] = calibration_vol_1(Maturities,Vol_shift,Strikes,disc_caplets,delta_caplets,dt_caplets,price,dates_caplets);
    totalVega_Spot_NPV = structurebond_spot(N,T,sf,s,s1,c1,c2,c3,dates_caplets,disc_caplets,spot_vol,Strikes,X)-NPV_spot;

end