function delta_NPV_spot = delta_bucket(datesSet,ratesSet,NPV_spot,X_spot,N,T,sf,s,s1,c1,c2,c3,dates_caplets,spot_vol,Strikes)

    % Computing the delta bucket for our portfolio made of the certificate
    %
    % datesSet  :   dates of market quoted products
    % ratesSet  :   rates of market quoted products
    % NPV_spot : NPV_spot (not exactly zero)
    % X_spot : percentage of the principal amount computed with spot
    %          volatilities
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
    % Strikes : volatility matrix Strikes
    
    actual365 = 3;
    
    % % Bootstrap for the basic case 
    % [dates, discounts]=bootstrap(datesSet, ratesSet);
    
    d_basis_point=1e-4;
    
    % Number of depos we are using for the bootstrap
    num_depos = find(datesSet.depos>datesSet.futures(1,1),1,"first");
    
    % Number of the liquid futures
    num_futures=7;      
    
    % Number of the swap used in our bootstrap (ie 49)
    num_swaps=length(ratesSet.swaps(:,1))-1;    
    
    delta_caplets=yearfrac(dates_caplets(1),dates_caplets(2:end),actual365);
    delta_NPV_spot=zeros(num_depos+num_futures+num_swaps,1);
    
    % Computing the delta-bucket for the depos
    for i=1:num_depos
        aux=ratesSet;
        aux.depos(i,:)=ratesSet.depos(i,:)+d_basis_point;
        
        [dates_shift, discounts_shift]=bootstrap(datesSet, aux);

        % pricing
        zerorates=zRates(dates_shift,discounts_shift);
        zeroRates_caplets=interp1(dates_shift,zerorates/100,dates_caplets(2:end));
        disc_caplets=exp(-zeroRates_caplets.*delta_caplets);
        delta_NPV_spot(i)=structurebond_spot(N,T,sf,s,s1,c1,c2,c3,dates_caplets,disc_caplets,spot_vol,Strikes,X_spot)-NPV_spot;
    end

% Computing the delta-bucket for the futures
    for i=1:num_futures
        aux=ratesSet;
        aux.futures(i,:)=ratesSet.futures(i,:)+d_basis_point;
        
        [dates_shift, discounts_shift]=bootstrap(datesSet, aux);
        % pricing
        zerorates=zRates(dates_shift,discounts_shift);
        zeroRates_caplets=interp1(dates_shift,zerorates/100,dates_caplets(2:end));
        disc_caplets=exp(-zeroRates_caplets.*delta_caplets);
        delta_NPV_spot(num_depos+i)=structurebond_spot(N,T,sf,s,s1,c1,c2,c3,dates_caplets,disc_caplets,spot_vol,Strikes,X_spot)-NPV_spot;
       
    end

% Computing the delta-bucket for the swaps
    for i=2:num_swaps
        aux=ratesSet;
        aux.swaps(i,:)=ratesSet.swaps(i,:)+d_basis_point;
        [dates_shift, discounts_shift]=bootstrap(datesSet, aux);
        % pricing
        zerorates=zRates(dates_shift,discounts_shift);
        zeroRates_caplets=interp1(dates_shift,zerorates/100,dates_caplets(2:end));
        disc_caplets=exp(-zeroRates_caplets.*delta_caplets);
        delta_NPV_spot(num_depos+num_futures+i-1)=structurebond_spot(N,T,sf,s,s1,c1,c2,c3,dates_caplets,disc_caplets,spot_vol,Strikes,X_spot)-NPV_spot;
    end
end