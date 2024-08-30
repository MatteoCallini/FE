function delta_NPV_spot = delta_bucket_cap(datesSet,ratesSet,cap_price,dates_caplets,sigma,Strike)

    % Computing the delta bucket for our portfolio made of the certificate
    %
    % datesSet  :   dates of market quoted products
    % ratesSet  :   rates of market quoted products
    % cap_price : Initial cap price
    % dates_caplets : caplets dates
    % sigma : spot volatilities vector
    % Strike : cap volatility
    
    actual365 = 3;
    actual360 = 2;
    
    d_basis_point=1e-4;
    
    % Number of depos we are using for the bootstrap
    num_depos = find(datesSet.depos>datesSet.futures(1,1),1,"first");
    
    % Number of the liquid futures
    num_futures=7;      
    
    % Number of the swap used in our bootstrap (ie 49)
    num_swaps=length(ratesSet.swaps(:,1))-1;    
    
    delta_caplets=yearfrac(dates_caplets(1),dates_caplets(2:end),actual365);
    delta_NPV_spot=zeros(num_depos+num_futures+num_swaps,1);
    dt_caplets = yearfrac(dates_caplets(1:end-1),dates_caplets(2:end),actual360); 

    % Computing the delta-bucket for the depos
    for i=1:num_depos
        aux=ratesSet;
        aux.depos(i,:)=ratesSet.depos(i,:)+d_basis_point;
        [dates_shift, discounts_shift]=bootstrap(datesSet, aux);
        % pricing
        zerorates = zRates(dates_shift,discounts_shift);
        zeroRates_caplets = interp1(dates_shift,zerorates/100,dates_caplets(2:end));
        disc_caplets_shifted = exp(-zeroRates_caplets.*delta_caplets);
        forward_disc_shifted = [disc_caplets_shifted(1), disc_caplets_shifted(2:end)./disc_caplets_shifted(1:end-1)];
        % Computing Libor rate
        Libor_shifted = (1./forward_disc_shifted-1)./dt_caplets;
        delta_NPV_spot(i) = Cap_pricing(sigma,Strike,disc_caplets_shifted,Libor_shifted,delta_caplets,dt_caplets) - cap_price;
    end

    % Computing the delta-bucket for the futures
    for i=1:num_futures
        aux=ratesSet;
        aux.futures(i,:)=ratesSet.futures(i,:)+d_basis_point;
        [dates_shift, discounts_shift]=bootstrap(datesSet, aux);
        % pricing
        zerorates = zRates(dates_shift,discounts_shift);
        zeroRates_caplets = interp1(dates_shift,zerorates/100,dates_caplets(2:end));
        disc_caplets_shifted = exp(-zeroRates_caplets.*delta_caplets);
        forward_disc_shifted = [disc_caplets_shifted(1), disc_caplets_shifted(2:end)./disc_caplets_shifted(1:end-1)];
        % Computing Libor rate
        Libor_shifted = (1./forward_disc_shifted-1)./dt_caplets;
        delta_NPV_spot(num_depos+i) = Cap_pricing(sigma,Strike,disc_caplets_shifted,Libor_shifted,delta_caplets,dt_caplets) - cap_price;
    end

    % Computing the delta-bucket for the swaps
    for i=2:num_swaps
        aux=ratesSet;
        aux.swaps(i,:)=ratesSet.swaps(i,:)+d_basis_point;
        [dates_shift, discounts_shift]=bootstrap(datesSet, aux);
        % pricing
        zerorates = zRates(dates_shift,discounts_shift);
        zeroRates_caplets = interp1(dates_shift,zerorates/100,dates_caplets(2:end));
        disc_caplets_shifted = exp(-zeroRates_caplets.*delta_caplets);
        forward_disc_shifted = [disc_caplets_shifted(1), disc_caplets_shifted(2:end)./disc_caplets_shifted(1:end-1)];
        % Computing Libor rate
        Libor_shifted = (1./forward_disc_shifted-1)./dt_caplets;
        delta_NPV_spot(num_depos+num_futures+i-1) = Cap_pricing(sigma,Strike,disc_caplets_shifted,Libor_shifted,delta_caplets,dt_caplets) - cap_price;
    end
end