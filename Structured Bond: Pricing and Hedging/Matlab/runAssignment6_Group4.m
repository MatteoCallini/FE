% runAssignment6
% group 4, AY2023-2024
%       
% to run:
% > runAssignment6_Group4

clear all
close all
clc
format long

actual360 = 2;      %depo day count
actual365 = 3;      %IB day count
thirty360 = 6;      %Swap day count

% fixing the seed
seed = 10;
rng(seed)

%% Settings
tic
formatData = 'dd/mm/yyyy';

%% Read market data
% This fuction works on Windows OS. Pay attention on other OS.
[datesSet, ratesSet] = readExcelData('MktData_CurveBootstrap_Assignment6_Group4.xls', formatData);

% taking the swap years alreay given 
swap_years_given = xlsread('MktData_CurveBootstrap_20-2-24.xls',1,'C39:C56');
fprintf("--- Point a ---\n")


%% Holidays for the next 50 years
% swap years from 1year to 50 years 
swap_years=1:swap_years_given(end);

% EU holidays 
EU=eurCalendar;
% copy of dataset
ratesSet_given=ratesSet;
datesSet_given=datesSet;

% taking the correspoding date for each year, considering holidays 
datesSet_given.swaps = datetime(datesSet.settlement, 'ConvertFrom', 'datenum') + calmonths(12*swap_years_given');     
datesSet_given.swaps = (datenum(busdate(datesSet_given.swaps-1,"follow",eurCalendar)))';

% taking the correspoding date for each year, considering holidays 
dates_swaps_given = datetime(datesSet.settlement, 'ConvertFrom', 'datenum') + calmonths(12*swap_years_given');     
datesSet.swaps = datetime(datesSet.settlement, 'ConvertFrom', 'datenum') + calmonths(12*swap_years'); 
dates_swaps_given = datenum(busdate(dates_swaps_given-1,"follow",eurCalendar));
datesSet.swaps = datenum(busdate(datesSet.swaps-1,"follow",eurCalendar));

% computing the mean between the bid and aks
mid_swap_given=(ratesSet.swaps(:,1) + ratesSet.swaps(:,2))/2;

% interpolation of the mean found before on the dates from 1y to 50y
mid_swap = interp1(yearfrac(datesSet.settlement,dates_swaps_given,actual365), ...
           mid_swap_given,...
           yearfrac(datesSet.settlement,datesSet.swaps,actual365),"spline")';

% subtituing bid and ask wjtih their mean (this is needed in order to give ratesSet as an input to the bootstrap function )
ratesSet.swaps=[mid_swap; mid_swap]';

%% Bootstrap
[dates, discounts] = bootstrap(datesSet, ratesSet);

%% Compute Zero Rates
zerorates = zRates(dates,discounts); % Given in "percentage". To obtain the value, you have to divide by 100

%% Plot Results
% discount curve
figure
plot(datetime(dates, 'ConvertFrom', 'datenum'),discounts,'-o')
title('Discount curve');

% zero-rates
figure
plot(datetime(dates(2:end), 'ConvertFrom', 'datenum'),zerorates(2:end),'-x')
xlim([datetime(dates(1), 'ConvertFrom', 'datenum')-300, datetime(dates(end), 'ConvertFrom', 'datenum')+300]);
title('Zero rates curve');
fprintf("Bootstrap computation time: ")
toc

%% Read Cap vol market data
fprintf("--- Point b ---\n")

% Raking strikes, maturities and volatilities from the excel table 
Strikes = xlsread('Caps_vol_20-2-24.xlsx',1,'F1:R1')/100; % Cap strikes
Maturities = xlsread("Caps_vol_20-2-24.xlsx",1,'B2:B16'); % Cap maturity
Volatilities = xlsread("Caps_vol_20-2-24.xlsx",1,'F2:R16')*1e-4; % Volatility matrix

%% Bootstrap of volatilities taken from the Cap

% Dates of the caplets (every quarter up to 30 years) with the settlement date
dates_caplets = datetime(datesSet.settlement,'ConvertFrom', 'datenum') + calmonths(3*(0:4*Maturities(end)));
dates_caplets = datenum(busdate(dates_caplets-1,"follow",eurCalendar));

zeroRates_caplets = interp1(dates,zerorates/100,dates_caplets(2:end));
delta_caplets = yearfrac(dates_caplets(1),dates_caplets(2:end),actual365); % from t0 to ti
discounts_caplets = exp(-zeroRates_caplets.*delta_caplets);
dt_caplets=yearfrac(dates_caplets(1:end-1),dates_caplets(2:end),actual360); % from ti to ti+1

%% Cap price for each volatility 
price = bachelier_price(Volatilities,Strikes,discounts_caplets,Maturities,delta_caplets,dt_caplets);

%% Bootstrap of the spot volatilities
tic
[spot_vol, spot_vol_mkt] = calibration_vol_1(Maturities,Volatilities,Strikes,discounts_caplets,delta_caplets,dt_caplets,price,dates_caplets);
price_spot = bachelier_price_spot(spot_vol,Strikes,discounts_caplets,Maturities,delta_caplets,dt_caplets);

% % Check
% diff_price = price_spot(:,1:length(Strikes)) - price;
% fprintf("Spot volatilities calibration time: ")

toc

%% Plot volatility surface
[X,Y]=meshgrid(Strikes',Maturities);
figure
mesh(X,Y,price);
title('3-D representation of the surface of prices')

[X,Y]=meshgrid(Strikes',Maturities);
figure
mesh(X,Y,Volatilities);
title('3-D representation of the surface of flat volatilities')

[X,Y]=meshgrid(Strikes',Maturities);
figure
mesh(X,Y,spot_vol_mkt);
title('3-D representation of the surface of spot volatilities')

%% Structured bond inizialization
N = 50*1e+6; % Notional
T = 15; % Maturity
s = 0.02; % fixed rate Bank
sf = 0.03; % first quarter coupon IB leg
s1 = 0.011; % fixed part of the IB leg
c1 = 0.043; % cap for the quarters up to 5 years - IB leg
c2 = 0.046; % cap for the quarters from 5 to 10 years - IB leg
c3 = 0.051; % cap for the quarters from 10 to 15 years - IB leg

delta_schedule = delta_caplets(1:(4*T));          % It starts from the settlement date
disc_schedule = discounts_caplets(1:(4*T));
dt_schedule = dt_caplets(1:(4*T));

% Computing forward discounts and forward rates 
forward_disc_schedule = [disc_schedule(1), disc_schedule(2:end)./disc_schedule(1:end-1)];

% Computing Libor rate
fwd_libor_schedule = (1./forward_disc_schedule-1)./dt_schedule;

%% Upfront and NPV
tic

% Notional percentage
X_spot = fzero(@(x) structurebond_spot(N,T,sf,s,s1,c1,c2,c3,dates_caplets,discounts_caplets,spot_vol,Strikes,x),1);
fprintf("Upfront (spot volatilities) in percentage: %f \n\n",X_spot*100)
fprintf("Upfront considering the notional equal to %d euros (spot volatilities): %f\n\n",N,X_spot*N)

% Npv check
NPV_spot = structurebond_spot(N,T,sf,s,s1,c1,c2,c3,dates_caplets,discounts_caplets,spot_vol,Strikes,X_spot);
fprintf("NPV considering the notional equal to %d euros (spot volatilities): %.10f\n\n",N,NPV_spot)
fprintf("NPV computation time: ")
toc

%% c) Delta-Bucket sensitivities
fprintf("---Point c---\n")

% This function computes the delta bucket considering the original
% bootstrap, before the swap spline interpolation

delta_NPV_spot = delta_bucket2(datesSet_given,ratesSet_given,NPV_spot,X_spot,N,T,sf,s,s1,c1,c2,c3,dates_caplets,spot_vol,Strikes,swap_years_given);
fprintf("Delta-Bucket sensitivity computation time: ")
tic
figure()
plot(1:length(delta_NPV_spot),delta_NPV_spot)
title('Delta-Bucket sensitivity spot caso 1')
xlabel('number of instrument')
ylabel('delta')
toc

% This function computes the delta bucket considering the
% bootstrap after the swap spline interpolation

tic
% delta bucket structured bond
delta_NPV_spot1 = delta_bucket(datesSet,ratesSet,NPV_spot,X_spot,N,T,sf,s,s1,c1,c2,c3,dates_caplets,spot_vol,Strikes);

fprintf("Delta-Bucket sensitivity computation time: ")
figure()
plot(1:length(delta_NPV_spot1),delta_NPV_spot1)
title('Delta-Bucket sensitivity spot caso 2')
xlabel('number of instrument')
ylabel('delta')
toc

%% d) Total-Vega sensitivity
fprintf("---Point d---\n")
tic
totalVega_Spot_NPV = total_vega(Maturities,Volatilities,Strikes,NPV_spot,N,T,sf,s,s1,c1,c2,c3,dates_caplets,discounts_caplets,X_spot);
fprintf("Total Vega: %.6f\n",totalVega_Spot_NPV )
fprintf("Total Vega sensitivity computation time: ")
toc

%% e) Vega-Bucket sensitivities
fprintf("---Point e---\n")
tic
vega_bucket_spot = vega_bucket(N,T,sf,s,s1,c1,c2,c3,dates_caplets,discounts_caplets,Volatilities,Maturities,Strikes,X_spot,NPV_spot);
for i=1:length(vega_bucket_spot)
    fprintf("Vega Bucket %dy: %.2f\n",Maturities(i),vega_bucket_spot (i))
end
fprintf("\nVega Bucket sum: %.3f\n",sum(vega_bucket_spot))
fprintf("Vega-bucket sensitivity computation time: ") % It is close to the Total Vega
toc
figure
plot(Maturities, vega_bucket_spot) 
title('Vega-Bucket Sensitivity ')
xlabel('Maturities')
ylabel('Vega')

%% f) Coarse-grained buckets
fprintf("---Point f---\n")

deltas = yearfrac(dates(1),dates(2:end),actual365);
[w2, w5, w10, w15] = weights_delta(deltas);

% %% Check sum weights==1 (for the relevant ones)
% w2 + w5 + w10 + w15

%% Plot
figure
plot(deltas,w2,'-o');
axis([0 20 0 1])
hold on 
plot(deltas,w5,'-o');

plot(deltas,w10,'-o');

plot(deltas,w15,'-o');
legend('w2','w5','w10','w15')

%% Coarse-Grained Delta
DV01_w2 = sum(w2.*delta_NPV_spot1);
DV01_w5 = sum(w5.*delta_NPV_spot1);
DV01_w10 = sum(w10.*delta_NPV_spot1);
DV01_w15 = sum(w15.*delta_NPV_spot1);

%% Sensitivities
tic
delta_bucket_swap_2y = Deltabucket_Swap_1(dates_caplets(1),dates_caplets(2:(4*2+1)),...
                       mean(ratesSet.swaps(2,:)),dates,discounts,datesSet,ratesSet);
delta_bucket_swap_5y = Deltabucket_Swap_1(dates_caplets(1),dates_caplets(2:(4*5+1)),...
                       mean(ratesSet.swaps(5,:)),dates,discounts,datesSet,ratesSet);
delta_bucket_swap_10y = Deltabucket_Swap_1(dates_caplets(1),dates_caplets(2:(4*10+1)),...
                       mean(ratesSet.swaps(10,:)),dates,discounts,datesSet,ratesSet);
delta_bucket_swap_15y = Deltabucket_Swap_1(dates_caplets(1),dates_caplets(2:(4*15+1)),...
                       mean(ratesSet.swaps(15,:)),dates,discounts,datesSet,ratesSet);
fprintf("Computation time for swap sensitivities: ")
toc

%% Coarse-Grained Bucket swap

% DV01_w15_swap2 = w15'*delta_bucket_swap_2y;
% DV01_w10_swap2 = w10'*delta_bucket_swap_2y;
% DV01_w5_swap2 = w5'*delta_bucket_swap_2y;
DV01_w2_swap2 = w2'*delta_bucket_swap_2y;

% DV01_w15_swap5 = w15'*delta_bucket_swap_5y;
% DV01_w10_swap5 = w10'*delta_bucket_swap_5y;
DV01_w5_swap5 = w5'*delta_bucket_swap_5y;
DV01_w2_swap5 = w2'*delta_bucket_swap_5y;

% DV01_w15_swap10 = w15'*delta_bucket_swap_10y
DV01_w10_swap10 = w10'*delta_bucket_swap_10y;
DV01_w5_swap10 = w5'*delta_bucket_swap_10y;
DV01_w2_swap10 = w2'*delta_bucket_swap_10y;

DV01_w15_swap15 = w15'*delta_bucket_swap_15y;
DV01_w10_swap15 = w10'*delta_bucket_swap_15y;
DV01_w5_swap15 = w5'*delta_bucket_swap_15y;
DV01_w2_swap15 = w2'*delta_bucket_swap_15y;

%% Hedging (Swap Payer) - Linear system 
tic
syms x y z t;

% We don't consider some weights since they are zero, you can check up
eqn1 = x*DV01_w15_swap15 + DV01_w15 == 0;
eqn2 = x*DV01_w10_swap15 + y*DV01_w10_swap10 + DV01_w10 == 0;
eqn3 = x*DV01_w5_swap15 + y*DV01_w5_swap10 + z*DV01_w5_swap5 + DV01_w5 == 0;
eqn4 = x*DV01_w2_swap15 + y*DV01_w2_swap10 + z*DV01_w2_swap5 + t*DV01_w2_swap2 + DV01_w2 == 0;
S = solve([eqn1,eqn2,eqn3,eqn4],[x y z t]); % it solves the linear system
X = double(S.x);
Y = double(S.y);
Z = double(S.z);
T1 = double(S.t);

fprintf("Linear system computation time: ");
toc
fprintf("Swap 15 y Notional: %.2f\n",X);
fprintf("Swap 10 y Notional: %.2f\n",Y);
fprintf("Swap 5 y Notional: %.2f\n",Z);
fprintf("Swap 2 y Notional: %.2f\n",T1);

% % If you remove the comments you can see what happens if we use the delta
% % bucket computed in point c), the case where the shift is done before 
% % the spline.
% 
% %% f) Coarse-grained buckets
% nDepos = find(datesSet.depos>datesSet.futures(1,1),1,"first");
% dates_bootstrap = [dates(1:(nDepos+7+1)),dates(nDepos+7+swap_years_given(2:end))];
% % datetime(dates_bootrsap, 'ConvertFrom', 'datenum')
% 
% deltas = yearfrac(dates_bootstrap(1),dates_bootstrap(2:end),actual365);
% [w2, w5, w10, w15] = weights_delta(deltas);
% 
% % %% Check sum weights==1 (for the relevant ones)
% % w2 + w5 + w10 + w15
% 
% %% Coarse-Grained Delta
% DV01_w2 = sum(w2.*delta_NPV_spot);
% DV01_w5 = sum(w5.*delta_NPV_spot);
% DV01_w10 = sum(w10.*delta_NPV_spot);
% DV01_w15 = sum(w15.*delta_NPV_spot);
% 
% %% Sensitivities
% tic
% delta_bucket_swap_2y = Deltabucket_Swap(dates_caplets(1),dates_caplets(2:(4*2+1)),...
%                        mean(ratesSet.swaps(2,:)),dates,discounts,datesSet,ratesSet,...
%                        swap_years_given,swap_years,eurCalendar);
% delta_bucket_swap_5y = Deltabucket_Swap(dates_caplets(1),dates_caplets(2:(4*5+1)),...
%                        mean(ratesSet.swaps(5,:)),dates,discounts,datesSet,ratesSet,...
%                        swap_years_given,swap_years,eurCalendar);
% delta_bucket_swap_10y = Deltabucket_Swap(dates_caplets(1),dates_caplets(2:(4*10+1)),...
%                        mean(ratesSet.swaps(10,:)),dates,discounts,datesSet,ratesSet,...
%                        swap_years_given,swap_years,eurCalendar);
% delta_bucket_swap_15y = Deltabucket_Swap(dates_caplets(1),dates_caplets(2:(4*15+1)),...
%                        mean(ratesSet.swaps(15,:)),dates,discounts,datesSet,ratesSet,...
%                        swap_years_given,swap_years,eurCalendar);
% fprintf("Computation time for swap sensitivities: ")
% toc
% 
% %% Coarse-Grained Bucket swap
% 
% DV01_w15_swap2 = w15'*delta_bucket_swap_2y
% DV01_w10_swap2 = w10'*delta_bucket_swap_2y
% DV01_w5_swap2 = w5'*delta_bucket_swap_2y
% DV01_w2_swap2 = w2'*delta_bucket_swap_2y
% 
% DV01_w15_swap5 = w15'*delta_bucket_swap_5y;
% DV01_w10_swap5 = w10'*delta_bucket_swap_5y;
% DV01_w5_swap5 = w5'*delta_bucket_swap_5y;
% DV01_w2_swap5 = w2'*delta_bucket_swap_5y;
% 
% DV01_w15_swap10 = w15'*delta_bucket_swap_10y;
% DV01_w10_swap10 = w10'*delta_bucket_swap_10y;
% DV01_w5_swap10 = w5'*delta_bucket_swap_10y;
% DV01_w2_swap10 = w2'*delta_bucket_swap_10y;
% 
% DV01_w15_swap15 = w15'*delta_bucket_swap_15y;
% DV01_w10_swap15 = w10'*delta_bucket_swap_15y;
% DV01_w5_swap15 = w5'*delta_bucket_swap_15y;
% DV01_w2_swap15 = w2'*delta_bucket_swap_15y;

%% g) Hedge the Vega with an ATM 5y Cap 
fprintf("---Point g---\n")

% All the computations for the caps are done both for a cap 5y which we
% need for this point g, but aldo for a cap 15y which will be asked in the
% next point h

% Cap maturities
Maturities_hedging = [5,15];
Maturities_dates = datetime(datesSet.settlement, 'ConvertFrom', 'datenum') + calmonths(12*Maturities');     
Maturities_dates = datenum(busdate(Maturities_dates-1,"follow",eurCalendar));

ATM_strikes = [(ratesSet_given.swaps(Maturities_hedging(1),1)+ratesSet_given.swaps(Maturities_hedging(1),2))/2,...
              (ratesSet_given.swaps(Maturities_hedging(2),1)+ratesSet_given.swaps(Maturities_hedging(2),2))/2];
sigma_ATM_spot = zeros(max(Maturities_hedging)*4-1,2);

% Spline interpolation to obtain the strikes
for j=1:length(ATM_strikes)
     for i=1:max(Maturities_hedging)*4-1
         sigma_ATM_spot(i,j) = interp1(Strikes, spot_vol(i,:), ATM_strikes(j), 'spline');
     end
end

% Cap prices
Cap_5y = Cap_pricing(sigma_ATM_spot(1:Maturities_hedging(1)*4-1,1),ATM_strikes(1),discounts_caplets(1:Maturities_hedging(1)*4),...
         fwd_libor_schedule(1:Maturities_hedging(1)*4),delta_caplets(1:(Maturities_hedging(1)*4-1)),dt_caplets(1:Maturities_hedging(1)*4));
Cap_15y = Cap_pricing(sigma_ATM_spot(1:Maturities_hedging(2)*4-1,2),ATM_strikes(2),discounts_caplets(1:Maturities_hedging(2)*4),...
          fwd_libor_schedule(1:Maturities_hedging(2)*4),delta_caplets(1:(Maturities_hedging(2)*4-1)),dt_caplets(1:Maturities_hedging(2)*4));

% we first compute the vega bucket of the products inside the portfolio 
% we already have the vega bucket of the certificate vega_bucket_spot
% we need the vega-bucket of the Cap 5y

%% Vega bucket sensitivities for the Caps
tic
vega_bucket_cap_5y = vega_bucket_spline(dates_caplets,discounts_caplets,Volatilities,Maturities,Strikes,...
                     ATM_strikes(1),5,Cap_5y,fwd_libor_schedule);
vega_bucket_cap_15y = vega_bucket_spline(dates_caplets,discounts_caplets,Volatilities,Maturities,Strikes,...
                      ATM_strikes(2),15,Cap_15y,fwd_libor_schedule);

fprintf("Computation time vega bucket Caps: ")
toc

% we put the delta bucket of the products inside the portfolio 
% we already have the delta-bucket for the certificate: delta_NPV_spot1
% (vector of 60 elements)
% we already have the delta-bucket for the swap 5y: delta_bucket_swap_5y
% (vector of 60 elements)
% we need the delta-bucket of the Cap

%% Delta bucket sensitivities for the Caps
% we already computed the delta bucket for 5 year cap

tic
delta_bucket_cap_5y = delta_bucket_cap(datesSet,ratesSet,Cap_5y,dates_caplets(1:5*4+1),sigma_ATM_spot(:,1),ATM_strikes(1));
delta_bucket_cap_15y = delta_bucket_cap(datesSet,ratesSet,Cap_15y,dates_caplets(1:15*4+1),sigma_ATM_spot(:,2),ATM_strikes(2));
fprintf("Delta bucket cap computation time")
toc

% Hedging the vega of the certificate by a 5y Cap ATM and
% then the delta for the portfolio (certificate+Cap) by a Swap 5y
[not_Cap,not_swap_5y]=hedging_vega(vega_bucket_spot,vega_bucket_cap_5y,delta_bucket_cap_5y,delta_NPV_spot1,delta_bucket_swap_5y);
fprintf("Cap 5y ATM Notional: %.2f\n",not_Cap);
fprintf("Swap 5y Notional: %.2f\n",not_swap_5y);

%% h) Vega DV01 Hedging
fprintf("---Point h---\n")

% Coarse-grained weights
deltas_vega = yearfrac(datesSet.settlement,Maturities_dates,actual365);
[w5_vega,w15_vega] = weights_vega(deltas_vega);

% % Check
% w5_vega + w15_vega

%% Vega Hedging
Vega_w5_bucket_5y = w5_vega'*vega_bucket_cap_5y;
% Vega_w15_bucket_5y = w15_vega'*vega_bucket_cap_5y;

Vega_w5_bucket_15y = w5_vega'*vega_bucket_cap_15y;
Vega_w15_bucket_15y = w15_vega'*vega_bucket_cap_15y;

% The linear system computes the hedging 
tic
syms x y;
eqn1 = x*Vega_w15_bucket_15y + w15_vega'*vega_bucket_spot  == 0;
eqn2 = x*Vega_w5_bucket_15y + y*Vega_w5_bucket_5y + w5_vega'*vega_bucket_spot  == 0;
S1 = solve([eqn1,eqn2],[x y]); % it solves the linear system
X1 = double(S1.x);
Y1 = double(S1.y);
fprintf("Linear system computation time: ");
toc

fprintf("Cap 15 y Notional: %.2f\n",X1);
fprintf("Cap 5 y Notional: %.2f\n",Y1);

%% Delta weights
[w5_delta_1, w15_delta_1] = weights_vega(deltas);

% % Check
% w5_delta_1 + w15_delta_1

%% Delta hedging
tic
syms x y;

% % Check
% w15_delta_1'*delta_bucket_cap_15y
% w15_delta_1'*delta_bucket_cap_5y
% w5_delta_1'*delta_bucket_swap_15y
% w5_delta_1'*delta_bucket_cap_5y

% We don't consider some weights since they are zero, you can check up
eqn1 = X1*w15_delta_1'*delta_bucket_cap_15y + x*w15_delta_1'*delta_bucket_swap_15y +...
       w15_delta_1'*delta_NPV_spot1== 0;
eqn2 = X1*w5_delta_1'*delta_bucket_cap_15y + Y1*w5_delta_1'*delta_bucket_cap_15y +...
       x*w5_delta_1'*delta_bucket_swap_15y +...
       y*w5_delta_1'*delta_bucket_swap_5y + w5_delta_1'*delta_NPV_spot1== 0;
S2 = solve([eqn1,eqn2],[x y]); % it solves the linear system
X2 = double(S.x);
Y2 = double(S.y);
fprintf("Swap 15 y Notional: %.2f\n",X2);
fprintf("Swap 5 y Notional: %.2f\n",Y2);
fprintf("Linear system computation time: ");
toc