function polar = c2p(cartesian)

%Function to transform cartesian coordinates into polar coordinates
%Written by Edmund Brekke July 2006
%@ cartesian:   Vectors in cartesian coordinates
%> polar:       Vectors in polar coordinates

%Get range using Pythagoras

r = sqrt(cartesian(1,:).^2 + cartesian(2,:).^2);

%Bearing is more difficult: All vectors must be checked separately

theta = zeros(size(r));
for(i=1:size(cartesian,2))

    %If x positive use arctangens

    if(cartesian(1,i) > 0)
        theta(i) = atan(cartesian(2,i)./cartesian(1,i));
    
    %If x negative use arctangens and add pi    
        
    elseif(cartesian(1,i) < 0)
        theta(i) = atan(cartesian(2,i)./cartesian(1,i)) + pi;

    else
        
        %If x zero and y negative bearing must be minus pi/2
        
        if(cartesian(2,i) < 0)
            theta(i) = -pi/2;
        
        %If x zero and y positive bearing must be plus pi/2    
            
        elseif(cartesian(2,i) > 0)
            theta(i) = pi/2;
            
        %If both zero set bearing to zero by default
            
        else
            theta(i) = 0;
        end
    end
end
polar = zeros(2,size(cartesian,2));
polar(1,:) = r;

%Make sure all bearings are in [0,2*pi)

polar(2,:) = mod(theta,2*pi);