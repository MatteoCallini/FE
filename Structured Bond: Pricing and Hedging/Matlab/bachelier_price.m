function price = bachelier_price(Volatilities,Strikes,discounts,Maturities,delta,dt)
    % This function computes the prices of every cap for the 3m Euribor (flat volatilities)
    % for every volatility
    %
    % Volatilities:     matrix of volalitilities from the table
    % Strikes:          vector of strikes 
    % discounts:        discounts taken from the bootstrap ( every 3 months) 
    % dates:            every 3 months
    % Maturities:       Cap maturities
    % delta:            time intervals referred always to t0
    % dt:               time intervals from t_i to t_(i+1)
    
    n = length(Maturities);
    m = length(Strikes);
    price = zeros(n,m);

    % Computing forward discounts and forward rates 
    forward_disc = [discounts(1), discounts(2:end)./discounts(1:end-1)];
    % Computing Libor rate
    Libor = (1./forward_disc-1)./dt; 

    for j=1:n % going through rows
        for i=1:m % through coloums 
            index = 4*Maturities(j);
            price(j,i) = Cap_pricing(Volatilities(j,i)*ones(size(Libor(1:index))),...
                                     Strikes(i),discounts(1:index),Libor(1:index),...
                                     delta(1:index),dt(1:index));
        end
    end
end
