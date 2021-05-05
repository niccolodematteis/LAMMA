%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       
%   matching (function of LAMMA.m)
%
%       v 2020.05.05
%
%   https://github.com/niccolodematteis/LAMMA.git
%
%       Niccolò Dematteis
%
%       This code is published under the
%       Licence CC BY-NC 4.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [DX,DY,DCC] = matching(refTile,searchTile,searchBand,varargin)
% function [DX,DY,DCC] = matching...
%   (refTile,searchTile,searchBand,options)
%
% INPUT:
%   refTile:        MxN reference patch
%   searchTile:     PxR reference patch P>=M; R>=N
%   searchBand:     1x4 positive integer array. Width of the search band in
%                   the directions [left right up down]
% OPTIONS:
%   oversampl:   numerical value of the subpixel sensitivity
%   subpixel:    Logical value. If true, it is caluclated a subpixel
%                displacement
%   method:      String that indicates the similarity function. zncc or
%                cosxcorr
%   step:        distance between consecutive points in the search band
%
% OUPUT:
%   DX          rightward motion of the refTile (px)
%   DY          downward motion of the refTile (px)
%   DCC         dcc matrix of the search area


defaultSubPixel = false;
defaultOverSmpl = 10;
defaultMethod = 'cosxcorr';
defaultStep = 1;

Opt = inputParser;
addRequired(Opt,'refTile',@isnumeric);
addRequired(Opt,'searchTile',@isnumeric);
addRequired(Opt,'searchBand',@isnumeric);
addParameter(Opt,'SubPixel',defaultSubPixel,@islogical);
addParameter(Opt,'OverSmpl',defaultOverSmpl,@isnumeric);
addParameter(Opt,'method',defaultMethod,@ischar);
addParameter(Opt,'step',defaultStep,@isnumeric);

parse(Opt,refTile,searchTile,searchBand,varargin{:});
SubPixel = Opt.Results.SubPixel;
OverSmplFactor = Opt.Results.OverSmpl;
method = Opt.Results.method;
step = Opt.Results.step;


[refRow,refCol]=size(refTile);
[searchRow,searchCol]=size(searchTile);

%Check inputs
CheckInputCorrectness;

%coordinates of the similarity matrix
[X,Y]=meshgrid(searchBand(1):searchBand(2),searchBand(3):searchBand(4));
%initialise the similarity matrix
DCC=zeros(size(X));


%define the similarity index functions
switch method
    %normalized cross-correlation 
    case 'zncc'
        A = refTile;
        A = A-mean(A(:),'omitnan');
        B = sum(sum(A.^2,'omitnan'),'omitnan');
        fun = @(A,B,Y) sum(sum( A.*(Y-mean(Y(:),'omitnan')),'omitnan'),'omitnan') / ...
            sqrt( B.* sum(sum( (Y-mean(Y(:),'omitnan')).^2,'omitnan'),'omitnan') );
    %cosine similarity correlation    
    case 'cosxcorr'
        [xg,yg]=gradient(refTile);
        A=sign(xg+1i*yg);
        [xg,yg]=gradient(searchTile);
        searchTile=sign(xg+1i*yg);
        B=0;
        fun = @(A,B,Y) mean(mean( real( conj(A).*Y ),'omitnan'),'omitnan') + B;
    otherwise
        error('unrecognised method')
end
%loop to compute similarity matrix
for ii=1:step:size(DCC,1)
    for jj=1:step:size(DCC,2)
        localTile = searchTile(ii:ii+refRow-1,jj:jj+refCol-1);
        DCC(ii,jj) = fun(A,B,localTile);
    end
end
if SubPixel
    %compute the subpixel displacement
    [DX,DY] = subpixOffsetMultiScale_nested();
else
    %identify the shift position
    [~,POS] = max(DCC(:));
    DY = Y(POS);
    DX = X(POS);
end

%--------------------------------------------------------------------------
    function [DxSubPix,DySubPix] = subpixOffsetMultiScale_nested()

% Adapted from
%
%    Debella-Gilo, M., & Kääb, A. (2011). 
%    Sub-pixel precision image matching for measuring surface displacements
%    on mass movements using normalized cross-correlation. 
%    Remote Sensing of Environment, 115(1), 130-142.        
%       
        RowOffset=searchBand(3);
        ColOffset=searchBand(1);
        
        %identify the shift position
        [~,POS] = max(DCC(:));
        [I,J] = ind2sub(size(DCC),POS);
        FirstApproxRow = I+RowOffset-1;
        FirstApproxCol = J+ColOffset-1;

        % pad of NaNs of DCC and then interp only on non-Nan values
        % try if 3x3 matrix is sufficient to interpolate 
        % otherwise use 5x5 matrix
        try
            %find the 3x3 area around the maximum of DCC
            I1 = I-1;
            I2 = I+1;
            J1 = J-1;
            J2 = J+1;
            %limits of the 3x3 area
            xi=-1;xf=1;
            yi=-1;yf=1;
            DCCnan = nanpadding(DCC,1);
            InterpArea = DCCnan(I1+1:I2+1,J1+1:J2+1);
            [JJ,II]=meshgrid(xi:xf,yi:yf);
            os = 1/OverSmplFactor;
            [nJJ,nII]=meshgrid(xi:os:xf,yi:os:yf);
            output=interp2(JJ,II,InterpArea,nJJ,nII,'cubic',nan);
        catch
            %find the 5x5 area around the maximum of DCC
            I1 = I-2;
            I2 = I+2;
            J1 = J-2;
            J2 = J+2;
            %limits of the 5x5 area
            xi=-2;xf=2;
            yi=-2;yf=2;
            DCCnan = nanpadding(DCC,2);
            InterpArea = DCCnan(I1+2:I2+2,J1+2:J2+2);
            [JJ,II]=meshgrid(xi:xf,yi:yf);
            os = 1/OverSmplFactor;
            [nJJ,nII]=meshgrid(xi:os:xf,yi:os:yf);
            output=interp2(JJ,II,InterpArea,nJJ,nII,'cubic',nan);
        end
        
        %evaluate the position of the maximum
        [~,POSgauss] = max(output(:));
        [relDY,relDX] = ind2sub(size(nJJ),POSgauss);
        subpixelDX = nJJ(relDY,relDX);
        subpixelDY = nII(relDY,relDX);
        %compute the actual shift
        DxSubPix = FirstApproxCol+subpixelDX;
        DySubPix = FirstApproxRow+subpixelDY;
                
    end
%--------------------------------------------------------------------------
    function CheckInputCorrectness
        
        if refRow>searchRow || refCol>searchCol
            error('Search tile must be greater than reference tile')
        end
        if numel(searchBand)~=4
            error('The search band must be a 4x1 array of positive integers')
        end
        if numel(OverSmplFactor)~=1 || OverSmplFactor<1 || mod(OverSmplFactor,1)~=0
            error('Oversamplingfactor must a positive integer')
        end
        if numel(step)~=1 || step<1
            error('step must be a positive integer')
        end
    end

end

%::::::::::::::::::::::::::::::::
function out = nanpadding(in,padsz)

[rw,cl]=size(in);

out = [nan(padsz,cl+2*padsz);
    nan(rw,padsz), in, nan(rw,padsz);
    nan(padsz,cl+2*padsz)];

end
