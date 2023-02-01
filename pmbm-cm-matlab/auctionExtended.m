function [ass,assItem,prices,val] = auctionExtended(RewardMatrix)

epsilon = 0.001;

% @A:       Gain matrix (positive gain better, dimension m x nTar)
% @epsilon: Tuning parameter. Should probably be updated online.
% @fa_flag: If true, then last row of A is expanded to facilitate multiple assignments to dummy measurement without altering the basic Auction algorithm
% >ass:     Assignment: Measurements for each track
% >assItem: Assignment: Tracks for each measurement
% >P:       Prices of each measurement

% Originally written July 2006 by Edmund Brekke
% Revised September 2016 to ....
% - Use a target-oriented formulation instead of measurement-oriented
% - Deal with misdetections through extension of gain matrix A
% Revision November 2017 by Edmund Brekke to fit with Orguners Murty method

A = RewardMatrix';
nTar = size(A,2);
m = size(A,1);
prices = zeros(m,1); % Declare price vector (one price for each measurements)
%epsilon = 0.1;

% The vector unassigned contains t's of all targets that haven't got a measurement assigned.

unassigned = (1:nTar)';

% The vector ass contains the final assignments, i.e. an i for each t.
% Now we want to allow i=N to be used several times.

ass = NaN*ones(nTar,1);
unassnumbers = nTar;

k = 0;
while ~isempty(unassigned)
    
    k = k + 1;

    
    
    %----------------------------------------------------------------------
    %Step 2----------------------------------------------------------------
    %----------------------------------------------------------------------
    
    %Pick out first unassigned target so that we can try to assign it
    
    unassOld = unassigned;
    
    t_current = unassigned(1);
    unassigned(1) = [];
    
    %----------------------------------------------------------------------
    %Step 3----------------------------------------------------------------
    %----------------------------------------------------------------------
    
    %Find tentative assignments for each track.
    %Do this by maximising gain minus price for each measurement
    
    ass_tentative = NaN*ones(nTar,1);
    for t=1:nTar
        
        possiblemeas = find(~isinf(A(:,t)));
        a = A(possiblemeas,t)-prices(possiblemeas);
        
        [ma,ix] = max(a);
        ass_tentative(t) = possiblemeas(ix);
        
    end
    
    
    
    %----------------------------------------------------------------------
    %Step 4----------------------------------------------------------------
    %----------------------------------------------------------------------
    
    %Find all tracks sharing preferred measurement of current track and add these tracks to the list of unassigned tracks
    
    i_current = ass_tentative(t_current);
    %if(i_current ~= m)
    
    
    q = find(ass == i_current);
    q(q==t_current) = [];
    
    for i=1:length(q)
        
        if(~isempty(q(i)) && isempty(find(unassigned == q(i),1)))
            
            ass(q(i)) = NaN;
            if(isempty(unassigned))
                unassigned = q(i);
            else
                unassigned = cat(1,unassigned,q(i));
            end
        end
        
    end
    %end
    
    if(length(unassigned) < 4 && length(unassOld) >=4)
        %k
    end
    
    %Assign preferred measurement of current track to current track
    
    ass(t_current) = i_current;
    
    %if(ass(4) == 1)
    %   error('eqeq');
    %end
    
    
    %----------------------------------------------------------------------
    %Step 5----------------------------------------------------------------
    %----------------------------------------------------------------------
    
    %Find all measurements feasible for t_current and their value for t_current
    
    possiblemeas = find(~isnan(A(:,t_current)));
    a = A(possiblemeas,t_current)-prices(possiblemeas);
    
    %Find the best and the second best measurement for t_current
    
    [ma1,ix1] = max(a);
    a(ix1) = a(ix1)*NaN;
    [ma2,ix2] = max(a);
    
    %Calculate how much t_current gains from choosing the best measurement instead of next best measurement and
    %increase price of best measurement correspondingly
    
    y = ma1-ma2;
    prices(i_current) = prices(i_current) + y + epsilon;
    
    %----------------------------------------------------------------------
    %Step 5----------------------------------------------------------------
    %----------------------------------------------------------------------
    
    unassnumbers = cat(1,unassnumbers,length(unassigned));
    

    
    
    if(k>=344477)
        
        error('whats going wrong in auction');
    end
end

%if(fa_flag)
%   ass(ass > mOld) = mOld; 
%end
assItem = zeros(m,1);
assItem(ass) = 1:nTar;
ass = ass';
assItem = assItem';

x = m2v([ ass; 1:nTar],size(A));
val = sum(A(x));
