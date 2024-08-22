function disc=interp1discount(b1,b2,d1,d2,dsett,d)

%interpolation of discount factor
%b1     discount factor i
%b2     discount factor i+1
%d1     time ti
%d2     time ti+1
%dsett  date settlement
%d      date were we need the interpolation
actual365=3;
    y1=zRates([dsett,d1,d2],[1,b1,b2])/100;
    y=interp1([d1,d2],y1(2:end),d);
    disc=exp(-(y*yearfrac(dsett,d,actual365)));
end