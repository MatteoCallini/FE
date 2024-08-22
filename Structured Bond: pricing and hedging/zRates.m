function zeroRates=zRates(dates,discounts)
%Function compute
    zeroRates(1)=0; %the first date is the settlment day, so the yield rate is equal to 0
    zeroRates = [zeroRates, -log(discounts(2:end))./yearfrac(dates(1),dates(2:end),3)]*100;
    % We write the zero rates in percentage, so we multiply by 100
    
end