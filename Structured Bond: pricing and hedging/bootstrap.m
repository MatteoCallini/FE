function [dates, discounts]=bootstrap(datesSet, ratesSet)

%function bootstrap taking dates and rates form Set in input

%datesSet           vector of all dates of depos, futures and swaps
%ratesSet           vector of all depos, futures and swaps rates
%dates              vector of dates for rates curve
%discounts          vector of discountes factor for our rates curve

actual360=2;
actual365=3;
thirty360=6;

    nDepos=length(datesSet.depos);
    nFutures=length(datesSet.futures);
    nSwaps=length(datesSet.swaps);

    % we find the mid price for every instruments
    deposRateMean=(ratesSet.depos(:,1)+ratesSet.depos(:,2))/2;
    futuresRateMean=(ratesSet.futures(:,1)+ratesSet.futures(:,2))/2;
    swapRateMean=(ratesSet.swaps(:,1)+ratesSet.swaps(:,2))/2;
 
    % we consider the settlment date    
    discounts(1)=1;
    dates(1)=datesSet.settlement;

%% depos
    % we consider the first expiry of the futures as the limit for the depos
    % expiries
    if sum(datesSet.depos==datesSet.futures(1,1))>0 % if the first future settlement date is in the depos dates 
        ldepos = find(datesSet.depos==datesSet.futures(1,1),1,'last')+1;  % we add 1 since in the vector of discout we have already put 1 in position 1
        discounts(2:ldepos)=1./(1+yearfrac(datesSet.settlement,datesSet.depos(1:ldepos-1),actual360).* ...
            deposRateMean(1:ldepos-1));
        dates(2:ldepos)=datesSet.depos(1:ldepos-1);
    else % if the first future settlement date is not in the depos dates 
        ldepos=find(datesSet.depos<datesSet.futures(1,1),1,'last')+2;  % we add 2 since in the vector of discout we have already put 1 in position 1 and we are considering the next one
        discounts(2:ldepos)=1./(1+yearfrac(datesSet.settlement,datesSet.depos(1:ldepos-1),actual360).* ...
            deposRateMean(1:ldepos-1));
        dates(2:ldepos)=datesSet.depos(1:ldepos-1);
    end

%% futures

    fdiscounts=1./(1+yearfrac(datesSet.futures(:,1),datesSet.futures(:,2),actual360).*futuresRateMean); %vector for fwd discount factor
    if sum(dates==datesSet.futures(1,1))==0   % we need interpolation to find the right discount factor
        i1=find(dates<datesSet.futures(1,1),1,'last');
        i2=find(dates>datesSet.futures(1,1),1,'first');
        d1=interp1discount(discounts(i1),discounts(i2),dates(i1),dates(i2),datesSet.settlement,datesSet.futures(1,1)); %interpolation
    else
        i1=find(dates==datesSet.futures(1,1)); % we don't need interpolation, we already have the right discout factor                               
        d1=discounts(i1);                                                               
    end
    discounts=[discounts, fdiscounts(1)*d1]; 
    dates=[dates, datesSet.futures(1:7,2)']; % we consider the first 7 futures
    
    for j=2:7
        if datesSet.futures(j,1)==datesSet.futures(j-1,2)
            discounts=[discounts, fdiscounts(j)*discounts(ldepos+j-1)];
        elseif datesSet.futures(j,1)<datesSet.futures(j-1,2)
            discAux=interp1discount(discounts(ldepos+j-2),discounts(ldepos+j-1), ...
                dates(ldepos+j-2),dates(ldepos+j-1),datesSet.settlement,datesSet.futures(j,1));
            discounts=[discounts, fdiscounts(j)*discAux];
        elseif datesSet.futures(j,1)>datesSet.futures(j-1,2)
            discounts=[discounts, fdiscounts(j)*discounts(ldepos+j-1)];
        end
    end

%% swaps
    % we find the first discount regarding the swaps
    i1=find(dates<datesSet.swaps(1),1,'last');
    i2=find(dates>datesSet.swaps(1),1,'first');
    d=interp1discount(discounts(i1),discounts(i2),dates(i1),dates(i2),datesSet.settlement,datesSet.swaps(1));
    
    % we find the second discount for the swap
    discounts=[discounts, (1-swapRateMean(2)* ...
            yearfrac(datesSet.settlement,datesSet.swaps(1),thirty360)*d)/...
            (1+yearfrac(datesSet.swaps(1),datesSet.swaps(2),thirty360)* ...
            swapRateMean(2))];
    d=[d, discounts(end)];
    dates=[dates, datesSet.swaps(2)];

    % swap discounts
    for j=3:nSwaps
        discounts=[discounts, (1-swapRateMean(j)* ...
            yearfrac([datesSet.settlement; datesSet.swaps(1:j-2)],datesSet.swaps(1:j-1),thirty360)'*d(1:j-1)')/...
            (1+yearfrac(datesSet.swaps(j-1),datesSet.swaps(j),thirty360)* ...
            swapRateMean(j))];
        dates=[dates, datesSet.swaps(j)];
        d=[d, discounts(end)];
    end
end