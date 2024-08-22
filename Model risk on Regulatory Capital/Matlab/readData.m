function data = readData(filename)
% Function to extract data about the rates from EIOPA saved in a .xls file
%
% INPUTS: 
% filename: Name of the file
%
% OUTPUT:
% data:    Struct with extracted data

% Extract years:
data.years = xlsread(filename,1,'A2:A38');
% Extract default probability for specaulative grade issuers:
data.DR_SG = xlsread(filename,1,'B2:B38');
% Extract default probability for all grade issuers:
data.DR_All_rated = xlsread(filename,1,'C2:C38');
% Extract recovery rates:
data.RR = xlsread(filename,1,'D2:D38');

end