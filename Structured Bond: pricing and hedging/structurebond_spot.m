function NPV = structurebond_spot(N,T,sf,s,s1,c1,c2,c3,dates_caplets,disc_caplets,Volatilities,Strikes,X)
    
    % This function computes the NPV of a structure bond with Euribor 3m
    % with maturity T
    %
    % N                 : Notional
    % T                 : Maturity
    % sf                : fixed coupon (1st quarter)
    % s                 : fixed part coupon of the fixed leg
    % s1                : fixed part coupon of the floating leg
    % c1                : cap for the quarters between 0.25-5 years floating coupon 
    % c2                : cap for the quarters between 5.25-10 years floating coupon 
    % c3                : cap for the quarters between 10.25-15 years floating coupon
    % dates_caplets     : caplets dates
    % disc_schedule     : discounts
    % Volatilities      : volatility matrix considering all the quarters
    % Strikes           : volatility matrix Strikes
    % X                 : percentage of the principal amount

    actual360 = 2;      %depo day count
    actual365 = 3;

    % dates of the schedule (every quarter up to T years) with the settlement date
    dates_schedule = dates_caplets(1:(4*T+1));
    disc_schedule = disc_caplets(1:(4*T));

    % zeroRates_schedule=interp1(dates,zerorates/100,dates_schedule(2:end));
    delta_schedule = yearfrac(dates_schedule(1),dates_schedule(2:end),actual365);
    % disc_schedule = exp(-zeroRates_schedule.*delta_schedule);
    dt_schedule = yearfrac(dates_schedule(1:end-1),dates_schedule(2:end),actual360);
    
    % Computing forward discounts and forward rates 
    forward_disc_schedule = [disc_schedule(1), disc_schedule(2:end)./disc_schedule(1:end-1)];
    % Computing Libor rate    
    fwd_libor = (1./forward_disc_schedule-1)./dt_schedule;
    
    % intepolating the volatilities in order to have the right strike
    idx1 = 4*T/3;
    idx2 = 4*2*T/3;
    idx3 = 4*T;

    % inizialiting the vector for the spot vol needed for the cap present
    % in the contract
    sigma1 = zeros(idx1-1,1);
    sigma2 = zeros(idx2-1,1);
    sigma3 = zeros(idx3-1,1);
    
    for i=1:idx1-1
        sigma1(i) = interp1(Strikes, Volatilities(i,:), c1-s1, 'spline');
    end
    for i=1:idx2-1
        sigma2(i) = interp1(Strikes, Volatilities(i,:), c2-s1, 'spline');
    end
    for i=1:idx3-1
        sigma3(i) = interp1(Strikes, Volatilities(i,:), c3-s1, 'spline');
    end

    % pricing the cap present in the contract with the spot vol
    cap1 = Cap_pricing(sigma1,c1-s1,disc_schedule(1:idx1),fwd_libor(1:idx1),delta_schedule(1:idx1),dt_schedule(1:idx1));
    cap2 = Cap_pricing(sigma2,c2-s1,disc_schedule(1:idx2),fwd_libor(1:idx2),delta_schedule(1:idx2),dt_schedule(1:idx2))...
           -Cap_pricing(sigma2(1:idx1-1),c2-s1,disc_schedule(1:idx1),fwd_libor(1:idx1),delta_schedule(1:idx1),dt_schedule(1:idx1));
    cap3 = Cap_pricing(sigma3,c3-s1,disc_schedule,fwd_libor,delta_schedule,dt_schedule)-...
           Cap_pricing(sigma3(1:idx2-1),c3-s1,disc_schedule(1:idx2),fwd_libor(1:idx2),delta_schedule(1:idx2),dt_schedule(1:idx2));
    
    NPV = N*(s*sum(disc_schedule.*dt_schedule)-... % yearfrac
          X-...
          sf*disc_schedule(1)*dt_schedule(1)-...
          s1*sum(disc_schedule(2:end).*dt_schedule(2:end))+...
          cap1+...
          cap2+...
          cap3+...
          fwd_libor(1)*dt_schedule(1)*disc_schedule(1));
end