function NPV = SWAPpricing(setDate,fixedLegPaymentDates,fixedRate,dates,discounts)
    
    % This function computes the price of a swap PAYER
    % setDate                    settlement date
    % fixedLegPaymentDates       dates when there payment of the fixed coupon
    % fixedrate                  fix rate
    % dates                      dates coming from bootstrap
    % discounts                  discounts coming from bootstrap
    
    thirty360=6;

    dt=yearfrac(setDate,fixedLegPaymentDates(1),thirty360);         %vector of year fraction for fix part
    dt=[dt,yearfrac(fixedLegPaymentDates(1:end-1),fixedLegPaymentDates(2:end),thirty360)']';   
    
    fixedLegdiscounts=interp1(dates,discounts,fixedLegPaymentDates);

    Pfix=fixedRate*sum(fixedLegdiscounts.*dt);
    Pfloat=1-fixedLegdiscounts(end);   %as we have seen in the lecture, we can simplify every floating leg since it is a telescopic sum
    NPV=Pfloat-Pfix;
end