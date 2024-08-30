function delta_NPV = Deltabucket_Swap_1(setDate,fixedLegPaymentDates,fixedRate,dates,discounts,datesSet,ratesSet)
    
    % This function computes the DV01 bucket for a Swap
    %
    % setDate                settlement date
    % fixedLegPaymentDates   Fixed payment dates
    % fixedRate              Swap rate
    % dates                  dates coming from the bootstrap
    % discounts              discounts factor coming from the original bootstrap
    % datesSet               initial Dates Set
    % ratesSet               initial Rates Set
    
    d_basis_point=1e-4;

    % Not exactly equal to zero
    NPV = SWAPpricing(setDate,fixedLegPaymentDates',fixedRate,dates,discounts);
    
    % Number of depos we are using for the bootstrap
    num_depos = find(datesSet.depos>datesSet.futures(1,1),1,"first");
    
    % Number of the liquid futures
    num_futures=7;      
    
    % Number of the swap used in our bootstrap (ie 49)
    num_swaps=length(ratesSet.swaps(:,1))-1;    
    delta_NPV=zeros(num_depos+num_futures+num_swaps,1);

    % Computing the delta-bucket for the depos
    for i=1:num_depos
        aux=ratesSet;
        aux.depos(i,:)=ratesSet.depos(i,:)+d_basis_point;
        [dates_shift, discounts_shift]=bootstrap(datesSet, aux);
        delta_NPV(i) = SWAPpricing(setDate,fixedLegPaymentDates',fixedRate,dates_shift,discounts_shift)-NPV;
    end

    % Computing the delta-bucket for the futures
    for i=1:num_futures
        aux=ratesSet;
        aux.futures(i,:)=ratesSet.futures(i,:)+d_basis_point;
        [dates_shift, discounts_shift]=bootstrap(datesSet, aux);
        delta_NPV(num_depos+i) = SWAPpricing(setDate,fixedLegPaymentDates',fixedRate,dates_shift,discounts_shift)-NPV;
    end

    % Computing the delta-bucket for the swaps
    for i=2:num_swaps
        aux=ratesSet;
        aux.swaps(i,:)=ratesSet.swaps(i,:)+d_basis_point;
        [dates_shift, discounts_shift]=bootstrap(datesSet, aux);
        delta_NPV(num_depos+num_futures+i-1) = SWAPpricing(setDate,fixedLegPaymentDates',fixedRate,dates_shift,discounts_shift)-NPV;
    end
end
