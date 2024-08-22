function [Add_on, Add_on_CI] = Add_on(RC,RC_naive,EL,EL_naive,RC_CI,EL_CI)

    % This function computes the Add on on the Regulatory capital after a
    % particular stress
    %
    % INPUTS
    % RC :            Regulatory capital
    % RC_naive :      Regulatory capital naive approach
    % EL :            Expected loss
    % EL_naive :      Expected loss naive approach
    % RC_CI :         Regulatory Capital
    %
    % OUTPUTS
    % Add_on :        Regulatory capital Add on ( Rc_new = RC*(1+Add_on) )
    % Add_on_CI :     Add on confidence interval

    Add_on = ((RC - RC_naive) + (EL - EL_naive))./RC_naive;
    Add_on_CI(:,1) = ((RC_CI(:,1) - RC_naive')+(EL_CI(:,1) - EL_naive'))./RC_naive';
    Add_on_CI(:,2) = ((RC_CI(:,2) - RC_naive')+(EL_CI(:,2) - EL_naive'))./RC_naive';
end