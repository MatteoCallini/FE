function delta_NPV_spot = delta_bucket2(datesSet,ratesSet,NPV_spot,X_spot,N,T,sf,s,s1,c1,c2,c3,dates_caplets,spot_vol,Strikes,swap_years_given)

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
        % swap_years_given: swap years given by the original bootstrap 
        
        actual365 = 3;
        
        % % Bootstrap for the basic case 
        % [dates, discounts]=bootstrap(datesSet, ratesSet);
        % 
        swap_years = 1:swap_years_given(end);
        % increment 
        d_basis_point = 1e-4;
        
        % Number of depos we are using for the bootstrap
        num_depos = find(datesSet.depos>datesSet.futures(1,1),1,"first");
        
        % Number of the liquid futures
        num_futures = 7;      
        
        % Number of the swap used in our bootstrap 
        num_swaps = length(ratesSet.swaps(:,1))-1;    
        
        delta_caplets = yearfrac(dates_caplets(1),dates_caplets(2:end),actual365);
        delta_NPV_spot = zeros(num_depos+num_futures+num_swaps,1);

        aux_dates=datesSet;
        aux_rates=ratesSet;
        % taking the correspoding date for each year, considering holidays 
        dates_swaps_given = datetime(datesSet.settlement, 'ConvertFrom', 'datenum') + calmonths(12*swap_years_given');     
        aux_dates.swaps = datetime(aux_dates.settlement, 'ConvertFrom', 'datenum') + calmonths(12*swap_years'); 
        dates_swaps_given = datenum(busdate(dates_swaps_given-1,"follow",eurCalendar));
        aux_dates.swaps = datenum(busdate(aux_dates.swaps-1,"follow",eurCalendar));

        % computing the mean between the bid and aks
        mid_swap_given=(aux_rates.swaps(:,1) + aux_rates.swaps(:,2))/2;
        % interpolation of the mean found before on the dates from 1y to 50y
        mid_swap = interp1(yearfrac(aux_dates.settlement,dates_swaps_given,actual365), ...
                   mid_swap_given,...
                   yearfrac(aux_dates.settlement,aux_dates.swaps,actual365),"spline")';

        % subtituing bid and ask wjtih their mean (this is needed in order to give ratesSet as an input to the bootstrap function )
        aux_rates.swaps=[mid_swap; mid_swap]';

        % Computing the delta-bucket for the depos
            for i = 1:num_depos

                aux_rates.depos(i,:) = aux_rates.depos(i,:)+d_basis_point;
 
                % Bootstrap
                [dates_shift, discounts_shift] = bootstrap(aux_dates, aux_rates);

                % pricing
                zerorates = zRates(dates_shift,discounts_shift);
                zeroRates_caplets = interp1(dates_shift,zerorates/100,dates_caplets(2:end));
                disc_caplets = exp(-zeroRates_caplets.*delta_caplets);

                delta_NPV_spot(i) = structurebond_spot(N,T,sf,s,s1,c1,c2,c3,dates_caplets,disc_caplets ...
                                                ,spot_vol,Strikes,X_spot)-NPV_spot;

                aux_rates.depos(i,:) = aux_rates.depos(i,:)-d_basis_point;

            end

        % Computing the delta-bucket for the futures
            for i = 1:num_futures

                aux_rates.futures(i,:) = aux_rates.futures(i,:)+d_basis_point;
                [dates_shift, discounts_shift] = bootstrap(aux_dates, aux_rates);
                % pricing
                zerorates = zRates(dates_shift,discounts_shift);
                zeroRates_caplets = interp1(dates_shift,zerorates/100,dates_caplets(2:end));
                disc_caplets = exp(-zeroRates_caplets.*delta_caplets);
                delta_NPV_spot(num_depos+i) = structurebond_spot(N,T,sf,s,s1,c1,c2,c3,dates_caplets,disc_caplets,spot_vol,Strikes,X_spot)-NPV_spot;
                aux_rates.futures(i,:) = aux_rates.futures(i,:)-d_basis_point;
            end
        
        % Computing the delta-bucket for the swaps
        % we add 1bp at each element of the original bootstrap, then we interpolate on the dates of the swap we need  
        for i = 2:num_swaps
            aux = ratesSet;
            aux_datesSet = datesSet;

            aux.swaps(i,:) = ratesSet.swaps(i,:)+d_basis_point;
            % holidays 
            EU=eurCalendar;
    
            % taking the correspoding date for each year, considering holidays 
            dates_swaps_given = datetime(aux_datesSet.settlement, 'ConvertFrom', 'datenum') + calmonths(12*swap_years_given');     
            aux_datesSet.swaps = datetime(aux_datesSet.settlement, 'ConvertFrom', 'datenum') + calmonths(12*swap_years'); 
            dates_swaps_given = datenum(busdate(dates_swaps_given-1,"follow",eurCalendar));
            aux_datesSet.swaps = datenum(busdate(aux_datesSet.swaps-1,"follow",eurCalendar));
            
            % computing the mean between the bid and aks
            mid_swap_given=(aux.swaps(:,1) + aux.swaps(:,2))/2;
            % interpolation of the mean found before on the dates from 1y to 50y
            mid_swap = interp1(yearfrac(aux_datesSet.settlement,dates_swaps_given,actual365), ...
                       mid_swap_given,...
                       yearfrac(aux_datesSet.settlement,aux_datesSet.swaps,actual365),"spline")';
            
            % subtituing bid and ask wjtih their mean (this is needed in order to give ratesSet as an input to the bootstrap function )
            aux.swaps=[mid_swap; mid_swap]';

            [dates_shift, discounts_shift]=bootstrap(aux_datesSet, aux);
            % pricing
            zerorates=zRates(dates_shift,discounts_shift);
            zeroRates_caplets=interp1(dates_shift,zerorates/100,dates_caplets(2:end));
            disc_caplets=exp(-zeroRates_caplets.*delta_caplets);
            delta_NPV_spot(num_depos+num_futures+i-1)=structurebond_spot(N,T,sf,s,s1,c1,c2,c3,dates_caplets,disc_caplets,spot_vol,Strikes,X_spot)-NPV_spot;
            
        end
end
