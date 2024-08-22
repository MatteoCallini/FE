function cap=Cap_pricing(sigma,Strike,discounts,Libor,delta,dt)

    % This function computes the value for a Cap with respect to the 3m Euribor
    %
    % sigma:        volatility. We consider a vector: in this way, if we have flat
    %               volatilities, we have a vector which contains the same sigma; if we
    %               have spot volatilities, we consider each one with its value 
    % Strike:       considered constant as well 
    % discounts:    taken from the bootstrap 
    % Libor:        forward libor computed with the discounts
    % delta:        time intervals referred always to t0
    % dt:           time intervals from t_i to t_(i+1)
    % 
    % we price the cap by summing the prices of the single caplets
    % we consider 4 caplets for each year and we don't consider the first
    % quarter caplet

    Caplet = zeros(length(Libor),1);
    for k=2:size(Libor,2)
        Caplet(k) = Pricing_Caplet(dt(k),discounts(k),Libor(k),Strike,sigma(k-1),delta(k-1));
    end
    cap=sum(Caplet);
end