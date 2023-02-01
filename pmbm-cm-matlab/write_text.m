function write_text(Y, figu)

% COPIED FROM NLE_LIMITED. MUST BE REVISED.

%function getpoint_cols_texs(Y, cnames,wsize)

%   dims =  selected dimensions to display from Y
%   Y   -   the low dimensional dataset
%   cnames - dataset from high dimensional patches
%   wsize - the size of small window

dims(1)=1;
dims(2)=2;

numDraw=0;

axx = axis;                     %   [xmin, xmax, ymin, ymax]
while 1,
    figure(figu) ; hold on ; 
    p = ginput(1);  %select one point from current axes
    
    if p(1)<axx(1), disp('OK') ; break ; end ; %quit if click outside axes on left
    if p(1)>axx(2), disp('OK') ; break ; end ; %quit if click outside axes on right
    if p(2)<axx(3), disp('OK') ; break ; end ; %quit if click outside axes on top
    if p(2)>axx(4), disp('OK') ; break ; end ; %quit if click outside axes on bottom
    
    dd = [Y(dims(1),:)-p(1);Y(dims(2),:)-p(2)] ; %difference between the screen coordinates and all points in the nominated dimensions of Y
    dd = ( dd(1,:).*dd(1,:) +dd(2,:).*dd(2,:)) ;
    [z,i] = min(dd) ; %find minimum distance to points
    
%     speaker=0;
%     while i>0
%        speaker=speaker+1;
%        i=i-nc;
%     end
%     words=i+nc
%     tmpstr=voice(ran(words,1):ran(words,2));
%     str=strcat(tmpstr,'-',num2str(speaker));
    
    str = num2str(i);
    
    gtext(str);
    %p2=ginput(1);
    %plot([p(1), p2(1)],[p(2),p2(2)]);
    
 end ;

%%%%%%%%%%%%%%%%%%   end of addition    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
return ;

