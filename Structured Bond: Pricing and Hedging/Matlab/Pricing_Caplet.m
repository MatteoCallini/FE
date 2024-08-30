function caplet=Pricing_Caplet(dt,discount,Libor,Strike,sigma,delta)

    % This function computes the price of caplet with the bachelier model from Ti to
    % Ti+1 with settlement in t0. We are considering the Norml Libor Market
    % model
    
    % dt             : yearfrac between Ti to Ti+1
    % discount       : discount from T0 to Ti+1
    % Libor          : libor rate between Ti to Ti+1
    % Strike         : strike of cap/caplet
    % sigma          : volatility (spot or flat)
    % delta          : yearfrac between T0 to Ti

    d = (Libor-Strike)./(sigma.*sqrt(delta));
    caplet = dt.*discount.*((Libor-Strike).*normcdf(d)+ ...
             sigma.*sqrt(delta).*normpdf(d));
    
end