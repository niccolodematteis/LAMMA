%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Local Adaptive Multiscale image Matching Algorithm (LAMMA)
%
%       version 1.0
%
%   https://github.com/niccolodematteis/LAMMA.git
%
%       Niccolò Dematteis
%       2021.10.13
%
%       This code is published under the
%       Licence CC BY-NC 4.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [DX,DY,NCC,nodes,calcNumber,BDlimits] = LAMMA...
    (MasterImage,SlaveImage,Parameters)
% [DX,DY,NCC,nodes,calcNumber,BDlim] = LAMMA(MasterImage,SlaveImage,tileSz,grid,options)
%
% INPUT:
%   MasterImage     MxN or MxNx3 real matrix
%   SlaveImage      MxN or MxNx3 real matrix
%   Parameters      Struct with fields:
%       tileSz          Positive integer. It is the side dimension of the
%                       squared patch where the image matching is applied
%       grid            Positive integer or Nx2 array. If it is a positive
%                       integer, it is the spatial resolution of the regular
%                       grid M/grid X N/grid where the image matching is computed
%       maxband         1X4 positive integer array. It is the dimension of the
%                       the search band in the directions [left, right, up,
%                       down] in the first scale
%       maxScale        Positive integer. If grid is a positive integer, it is
%                       the spatial resolution of the coarsest regular grid. If
%                       grid is an array, it is the number of nodes of the
%                       first irregular grid.
%   Optional Parameters
%       oversampling    oversampling factor for subpixel offset
%       neigh:          Positive integer. It is the number of the neighbours
%                       that are considered to adjust the interrogation area.
%                       Default is 4
%                       This option is valid only when using sparse grids
%       tolerance       Null or positive integer. It is the tolerance term that
%                       is added to the interrgoation area - Default is 2
%       Method          Similarity functions: ZNCC or COSXCORR - Default is
%                       COSXCORR
%       Seeds           NX2 array. Seeds contains the coordinates of the nodes
%                       that must be added to the first grid. Default is
%                       empty
%       printInfo       Logical value. If true, some information on the
%                       processing are displayed during the run - Default
%                       is false
% OUTPUTS:
%   DX, DY          Cell array of the horizontal and vertical offsets
%                   obtained with the image matching. Horizontal offsets
%                   are positive rightward. Vertical offsets are positive
%                   downward. Every cell contains the offsets of a single
%                   level
%   NCC             Cell array of the similarity index. Every cell contains
%                   the offsets of a single level
%   nodes           Cell array of the nodes coordinates of every level. The
%                   coordinates are a Nx2 matrix with [column,row] indexes
%   calcNumber      Array of the computational complexity of every level
%   BDlimits        Struct array of the search band limits of every node
%                   BDlimits.yM upper limits
%                   BDlimits.yP lower limits
%                   BDlimits.xM right limits
%                   BDlimits.xP left limits
%
%--------------------------------------------------------------------------

%check input
[MasterImage,SlaveImage,Parameters]=checkInput(MasterImage,SlaveImage,Parameters);
%get parameters
tileSz=Parameters.tileSz;
%for simplicity, take half the tile size
tileSz=floor(tileSz/2);
grid=Parameters.grid;
maxScale=Parameters.maxScale;
maxband=Parameters.maxband;
neigh=Parameters.neigh;
tolerance=Parameters.tolerance;
method=Parameters.Method;
seeds=Parameters.seeds;
printInfo=Parameters.printInfo;
os=Parameters.oversampling;
regularGrid=Parameters.RegularGrid;
%prepare processors pool
Pool=load_parpool;

%====================  SCALES  ================================
%create the grids of every scale

if regularGrid %REGULAR CASE
    nodeDist=grid; %finest resolution
    %calculating the various spatial resolutions
    %start from the coarsest one
    step=maxScale;
    count=0;
    while step>=nodeDist
        count=count+1;
        vec(count)=step;
        step=step/2;
    end
    %sort in ascending the resolutions and round to integer values
    vec=sort(round(vec));
    %if the first element is greater than nodeDist then I will add nodeDist
    %to the array
    if vec(1)>nodeDist
        vec=[nodeDist;vec(:)];
    end
    %take the number of scales
    numScales=numel(vec);
    %determine the nodes of every scale
    for ii=numScales:-1:1
        step=vec(ii);
        [rw,cl]=size(MasterImage);
        [X,Y]=meshgrid(1:step:cl,1:step:rw);
        %if there are manual seeds, they are added to the coarsest grid
        if ii==numScales && ~isempty(seeds)
            nodes{ii}=[X(:),Y(:);seeds(:,1),seeds(:,2)];
        else
            nodes{ii}=[X(:),Y(:)];
        end
    end
    %exclude the nodes already present in previous levels
    n1=nodes{numel(vec)};
    for ii=numel(vec)-1:-1:1
        n2=nodes{ii};
        pun=ismember(n2,n1,'rows');
        nodes{ii}=n2(~pun,:);
        n1=[n1;n2];
    end
    %nodes is a cell array with Nx2 array of evenly distributed grids
    %without common nodes
    
else %IRREGULAR CASE
    centroidNumber=maxScale; %number of nodes of the first scale
    count=0;
    while size(grid,1)>centroidNumber
        count=count+1;
        %take the indexes of the centroids and add them to the current grid
        [~,~,~,~,midx] = ...
            kmedoids(grid,centroidNumber,'Options',statset('UseParallel',true));
        %if there are manual seeds, they are added to the coarsest grid
        if count==1 && ~isempty(seeds)
            nodes{count}=[grid(midx,1),grid(midx,2);seeds(:,1),seeds(:,2)];
        else
            nodes{count}=[grid(midx,1),grid(midx,2)];
        end
        %remove centroids before to continue with the subsequent grids
        grid(midx,:)=[];
        %now use twice the number of nodes than in the previous grid
        centroidNumber=centroidNumber*3;
        %if there are too few remaining nodes, they are all included in the
        %last grid
    end
    if size(grid(:,1))<=centroidNumber
        count=count+1;
        nodes{count}=[grid(:,1),grid(:,2)];
    end
    %nodes is a cell array with Nx2 array of sparse grids
    %without common nodes
    
    %take the number of scales
    numScales=numel(nodes);
end
%determine the first neighbours of every node, considering the previous
%levels. The neighbours are used to adjust the search band limits in
%the
if license('test','Statistics_Toolbox')
    for ii=numScales-1:-1:1
        neighbour{ii}=...
            knnsearch(vertcat(nodes{ii+1:numScales}),nodes{ii},'K',neigh,...
            'includeties',true);
    end
else
    for ii=numScales-1:-1:1
        dat1=vertcat(nodes{ii+1:numScales});
        dat1=dat1(:,1)+1i*dat1(:,2);
        dat2=nodes{ii};
        dat2=dat2(:,1)+1i*dat2(:,2);
        M=nan(numel(dat1),4);
        parfor c=1:numel(dat2)
            [~,J] = mink( abs(dat2(c)-dat1),4 );
            M(c,:)=J;
        end
        neighbour{ii}=M;
    end
end
neighbour{numScales}=[];%the nodes of the first scale do not have neighbours
%====================  SCALES ENDS  ================================

if printInfo
    disp('Processing starts at')
    disp(datestr(now))
end

%loops on the various scales, starting from the coarsest grid
for level=numScales:-1:1
    %take the coordinates of the current level
    X=nodes{level}(:,1);
    Y=nodes{level}(:,2);
    if level==numScales
        %in the first scale, the search band limits are those imposed by
        %the user for all the nodes
        Xbm=ones(numel(X),1)*maxband(1);
        Xbp=ones(numel(X),1)*maxband(2);
        Ybm=ones(numel(X),1)*maxband(3);
        Ybp=ones(numel(X),1)*maxband(4);
    else
        %====================  ADAPTATION  ================================
        %reset the variables
        Ybm=[];
        Ybp=[];
        Xbm=[];
        Xbp=[];
        tmpN=[NCC{level+1:numScales}];
        tmpX=[DX{level+1:numScales}];
        tmpY=[DY{level+1:numScales}];
        for i=1:numel(X)
            %take the indexes of the neighbours of every node
            idx=neighbour{level}(i,:);
            idx=idx{1};
            %the similarity index is nan if there are missing data. Then,
            %use the initial search band
            if all(isnan(tmpN(idx)))
                Ybm(i)=maxband(3);
                Ybp(i)=maxband(4);
                Xbm(i)=maxband(1);
                Xbp(i)=maxband(2);
            else
                %if there are more than 2 neighbours I remove the first
                %fourth with the lowest ncc
                if sum(~isnan(tmpN(idx)))>2
                    idx=idx(~isnan(tmpN(idx)));
                    [~,reliable]=sort(tmpN(idx));
                    rmv=round(numel(reliable)/4);
                    idx(reliable(1:rmv))=[];
                end
                %add the tolerance term to the band limits
                Ybm(i)=min(tmpY(idx))-tolerance;
                Ybp(i)=max(tmpY(idx))+tolerance;
                Xbm(i)=min(tmpX(idx))-tolerance;
                Xbp(i)=max(tmpX(idx))+tolerance;
            end
        end
    end
    %Ybm, Ybp, Xbm and Xbp are the limits of the search band in the
    %directions [down,up,left,right]
    %====================  ADAPTATION ENDS  ================================
    
    %reset these variable
    dx=[];
    dy=[];
    ncc=[];
    void=[];
    
    timeCalc=zeros(numel(X),1);
    n=tic;
    %if the parallel computing toolbox is available the for loop is run in
    %parallel using half the cores available on the computer
    %parfor (jj=1:numel(X),opts)
    parfor jj=1:numel(X)
        x0=X(jj);
        y0=Y(jj);
        %take the search area limits for the current node
        rm=round(Ybm(jj));
        rp=round(Ybp(jj));
        cm=round(Xbm(jj));
        cp=round(Xbp(jj));
        %check if the search band goes outside the image. If it occurs,
        %the image matching results are set to NaN
        tol = tolerance*(numScales-level+1)+1;
        if y0-tileSz+maxband(3)-tol<1 || ...
                y0+tileSz+maxband(4)+tol>size(SlaveImage,1) || ...
                x0-tileSz+maxband(1)-tol<1 || ...
                x0+tileSz+maxband(2)+tol>size(SlaveImage,2)
            dx(jj)=nan;
            dy(jj)=nan;
            ncc(jj)=nan;
            void(jj)=1;
        else
            void(jj)=0;
            %take the reference patch
            refTile=MasterImage(y0-tileSz:y0+tileSz,x0-tileSz:x0+tileSz);
            %take the slave patch. The limits depends on the interrogation
            %area limits
            searchTile=SlaveImage(y0-tileSz+rm:y0+tileSz+rp,...
                x0-tileSz+cm:x0+tileSz+cp);
            %check if the patches are not NaN and if they have not constant
            %values. If it occurs, the results are set to NaN
            if all(refTile(:)==refTile(1,1)) || all(searchTile(:)==searchTile(1,1))
                dx(jj)=nan;
                dy(jj)=nan;
                ncc(jj)=nan;
                void(jj)=1;
            else
                %====================  MATCHING  ================================
                t=tic;
                %calculate the similarity function
                [dx(jj),dy(jj),DCC] = matching(refTile,searchTile,[cm,cp,rm,rp],os,method)        
                ncc(jj)=max(DCC(:));
                timeCalc(jj)=toc(t);
                %====================  MATCHING ENDS ============================
            end
        end
    end
    calcNumber(level)= sum( (abs(round(Ybm(void==0))-round(Ybp(void==0)))+1).*...
        (abs(round(Xbm(void==0))-round(Xbp(void==0)))+1) );
    cycleTime=toc(n);
    overheadTime=cycleTime-sum(timeCalc,'omitnan')/Pool;
    if printInfo
        disp(' ')
        disp(['Number of calculi: ',...
            num2str(calcNumber(level))])
        disp(['Iteration time: ',num2str(cycleTime)])
        disp(['Overhead time: ',num2str(overheadTime)])
        disp(['# operations/s ',...
            num2str(calcNumber(level)/(sum(timeCalc,'omitnan')/Pool))])
    end
    %store the results into a cell array. Every cell contains the results
    %of one scale
    DX{level}=dx;
    DY{level}=dy;
    NCC{level}=ncc;
    %store the band limits into a structure array
    BDlimits(level).yM=Ybm;
    BDlimits(level).yP=Ybp;
    BDlimits(level).xM=Xbm;
    BDlimits(level).xP=Xbp;
end


if printInfo
    disp('Processing ends at')
    disp(datestr(now))
end


end

%(((((((((((((((((((((((((((((((FUNCTIONS))))))))))))))))))))))))))))))))))
function [MasterImage,SlaveImage,Parameters]=...
    checkInput(MasterImage,SlaveImage,Parameters)

if any(size(MasterImage)~=size(SlaveImage))
    error('Master and slave images must have the same size')
end
%convert input data in single format
if ~isa(MasterImage,'single') || ~isa(SlaveImage,'single')
    MasterImage=single(MasterImage);
    SlaveImage=single(SlaveImage);
end
%convert images in monochromatic scale if necessary
if size(MasterImage,3)>1
    MasterImage=mean(MasterImage,3);
end
if size(SlaveImage,3)>1
    SlaveImage=mean(SlaveImage,3);
end
%substitute nan with zeros to speedup check within parfor
MasterImage(isnan(MasterImage))=0;
SlaveImage(isnan(SlaveImage))=0;

%check options
if numel(Parameters.grid)==1
    Parameters.RegularGrid=true;
    Parameters.neigh=4;
else
    Parameters.RegularGrid=false;
    if ~license('test','Statistics_Toolbox')
        error('The Statistics and Machine Learning Toolbox is required for irregular nodes')
    elseif ~isfield(Parameters,'neigh')
        Parameters.neigh=4;
    end
end
if ~isfield(Parameters,'tolerance')
    Parameters.tolerance=2;
end
if ~isfield(Parameters,'Method')
    Parameters.Method='cosxcorr';
end
if ~isfield(Parameters,'seeds')
    Parameters.seeds=[];
end
if ~isfield(Parameters,'printInfo')
    Parameters.printInfo=false;
end
if ~isfield(Parameters,'oversampling')
    Parameters.os=10;
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function Pool=load_parpool
%check whether the parallel computing toolbox is installed
%if yes, create a pool of half the avaliable cores
if license('test','Distrib_Computing_Toolbox')
    maxcores = str2double(getenv('NUMBER_OF_PROCESSORS'));
    pool=gcp('nocreate');
    if ~isempty(pool) && pool.NumWorkers~=round(maxcores/2)
        delete(pool)
        Pool=parpool(round(maxcores/2));
        Pool=Pool.NumWorkers;
    elseif isempty(pool)
        Pool=parpool(round(maxcores/2));
        Pool=Pool.NumWorkers;
    else
        Pool=pool.NumWorkers;
    end
else
    Pool=1;
end
%the following command distributes in blocks the nodes among the cores
%this should be useful if there are many nan or zero patches
% opts = parforOptions(gcp('nocreate'),'RangePartitionMethod','fixed',...
%     'SubrangeSize',1000);
end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function [DX,DY,DCC] = matching(refTile,searchTile,searchBand,OverSmplFactor,method)
%
% INPUT:
%   refTile:        MxN reference patch
%   searchTile:     PxR reference patch P>=M; R>=N
%   searchBand:     1x4 positive integer array. Width of the search band in
%                   the directions [left right up down]
%   oversampl:      numerical value of the subpixel sensitivity
%   method:         String that indicates the similarity function. zncc or
%                   cosxcorr
%
% OUPUT:
%   DX          rightward motion of the refTile (px)
%   DY          downward motion of the refTile (px)
%   DCC         dcc matrix of the search area

[refRow,refCol]=size(refTile);
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
[row,col]=size(DCC);
parfor ii=1:row
    for jj=1:col
        localTile = searchTile(ii:ii+refRow-1,jj:jj+refCol-1);
        DCC(ii,jj) = fun(A,B,localTile);
    end
end

if OverSmplFactor==1
    %identify the shift position
    [~,POS] = max(DCC(:));
    DY = Y(POS);
    DX = X(POS);
else
    %compute the subpixel displacement
    [DX,DY] = subpixOffset(DCC,searchBand,OverSmplFactor);
end

end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function [DxSubPix,DySubPix] = subpixOffset(DCC,searchBand,OverSmplFactor)

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

%--------------------------------------------------------------------------
function out = nanpadding(in,padsz)

[rw,cl]=size(in);

out = [nan(padsz,cl+2*padsz);
    nan(rw,padsz), in, nan(rw,padsz);
    nan(padsz,cl+2*padsz)];

end
%--------------------------------------------------------------------------


