function delta_NPV = Deltabucket_Swap(setDate,fixedLegPaymentDates,fixedRate,dates,discounts,datesSet,ratesSet,swap_years_given,swap_years,eurCalendar)

    % This function computes the DV01 bucket for a Swap
    %
    % setDate                settlement date
    % fixedLegPaymentDates   Fixed payment dates
    % fixedRate              Swap rate
    % dates                  dates coming from the bootstrap
    % discounts              discounts factor coming from the original bootstrap
    % datesSet               initial Dates Set
    % ratesSet               initial Rates Set
    % swap_years_given       initial swap years
    % swap_years             total swap years
    % eurCalendar            EU holidays

    actual365 = 3;
    d_basis_point=1e-4;

    % Not exactly equal to zero
    NPV = SWAPpricing(setDate,fixedLegPaymentDates',fixedRate,dates,discounts);
    
    % Number of depos we are using for the bootstrap
    num_depos = find(datesSet.depos>datesSet.futures(1,1),1,"first");
    
    % Number of the liquid futures
    num_futures=7;      
    
    % Number of the swap used in our bootstrap (ie 49)
    num_swaps=length(swap_years_given)-1;    
    aux_rates = ratesSet.swaps(swap_years_given,1);
    delta_NPV = zeros(num_depos+num_futures+num_swaps,1);
    % Computing the delta-bucket for the depos
    for i=1:num_depos
        aux=ratesSet;
        aux.depos(i,:) = ratesSet.depos(i,:)+d_basis_point;
        [dates_shift, discounts_shift]=bootstrap(datesSet, aux);
        % pricing
        delta_NPV(i) = SWAPpricing(setDate,fixedLegPaymentDates',fixedRate,dates_shift,discounts_shift)-NPV;
    end

    % Computing the delta-bucket for the futures
    for i=1:num_futures
        aux=ratesSet;
        aux.futures(i,:)=ratesSet.futures(i,:)+d_basis_point;
        [dates_shift, discounts_shift]=bootstrap(datesSet, aux);
        % pricing
        delta_NPV(num_depos+i) = SWAPpricing(setDate,fixedLegPaymentDates',fixedRate,dates_shift,discounts_shift)-NPV;
    end

    for i = 1:num_swaps

        aux_datesSet = datesSet;
        aux = ratesSet;
        aux_rates(i+1,:) = aux_rates(i+1,:)+d_basis_point;

        % taking the correspoding date for each year, considering holidays 
        dates_swaps_given = datetime(aux_datesSet.settlement, 'ConvertFrom', 'datenum') + calmonths(12*swap_years_given');     
        aux_datesSet.swaps = datetime(datesSet.settlement, 'ConvertFrom', 'datenum') + calmonths(12*swap_years'); 
        dates_swaps_given = datenum(busdate(dates_swaps_given-1,"follow",eurCalendar));
        aux_datesSet.swaps = datenum(busdate(datesSet.swaps-1,"follow",eurCalendar));
        
        mid_swap = interp1(yearfrac(aux_datesSet.settlement,dates_swaps_given,actual365), ...
                   aux_rates,...
                   yearfrac(aux_datesSet.settlement,aux_datesSet.swaps,actual365),"spline")';

        % subtituing bid and ask wjtih their mean (this is needed in order to give ratesSet as an input to the bootstrap function )
        aux.swaps=[mid_swap; mid_swap]';
        [dates_shift, discounts_shift]=bootstrap(aux_datesSet, aux);

        % pricing
        delta_NPV(num_depos+num_futures+i) = SWAPpricing(setDate,fixedLegPaymentDates',fixedRate,...
                                             dates_shift,discounts_shift)-NPV;
        aux_rates(i,:) = aux_rates(i,:) - d_basis_point;
    end
end