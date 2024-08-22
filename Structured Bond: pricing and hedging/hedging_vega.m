function [not_Cap,not_swap_5y]=hedging_vega(vega_bucket_spot,vega_bucket_cap_5y,delta_bucket_cap_5y,delta_NPV_spot1,delta_bucket_swap_5y)
    
    % The function hedge the vega of the certificate by a 5y Cap ATM and
    % then the delta for the portfolio (certificate+Cap) by a Swap 5y
 
    num_depos=4;
    num_futures=7;
    num_swaps=49;

    % In order to hedge for the 5th year we consider the corresponding bucket 
    % this index allow us to get the right bucket in the vega_bucket
    index_vega=5;

    % vega of the certificate + vega of the CAP *notional===0
    f1=@(x) vega_bucket_spot(index_vega)+vega_bucket_cap_5y(index_vega)*x;
    not_Cap=fzero(f1,100);

    % this index allow us to get the right bucket in the delta_bucket
    index_delta=num_depos+num_futures+4;
   
    f2=@(x) delta_NPV_spot1(index_delta)+...
                    delta_bucket_cap_5y(index_delta)*delta_bucket_cap_5y(index_delta)...
                    +x*delta_bucket_swap_5y(index_delta);
    not_swap_5y=fzero(f2,10000);
end