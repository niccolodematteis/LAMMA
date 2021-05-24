%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       
%   Local Adaptive Multiscale image Matching Algorithm (LAMMA)
%       
%       v 2020.05.24
%
%   https://github.com/niccolodematteis/LAMMA.git
%
%       Niccolò Dematteis
%
%       This code is published under the
%       Licence CC BY-NC 4.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [DX,DY,NCC,nodes,calcNumber,BDlimits] = LAMMA...
    (MasterImage,SlaveImage,tileSz,grid,varargin)
% [DX,DY,NCC,nodes,calcNumber,BDlim] = LAMMA(MasterImage,SlaveImage,tileSz,grid,options)
%    
% INPUT:
%   MasterImage     MxN or MxNx3 real matrix
%   SlaveImage      MxN or MxNx3 real matrix
%   tileSz          Positive integer. It is the side dimension of the 
%                   squared patch where the image matching is applied
%   grid:           Positive integer or Nx2 array. If it is a positive
%                   integer, it is the spatial resolution of the regular 
%                   grid M/grid X N/grid where the image matching is computed
% OPTIONS:
%   neigh:          Positive integer. It is the number of the neighbours
%                   that are considered to adjust the interrogation area.
%                   This option is valid only when using sparse grids
%   oversampling    oversamplign factor for subpixel displacement
%                   sensitivity
%   tolerance       Null or positive integer. It is the tolerance term that
%                   is added to the interrgoation area
%   maxband         1X4 positive integer array. It is the dimension of the 
%                   the search band in the directions [left, right, up,
%                   down] in the first scale
%   subpixel        Logical value. If true, the image matching is
%                   calculated with subpixel sensitivity
%   maxScale        Positive integer. If grid is a positive integer, it is
%                   the spatial resolution of the coarsest regular grid. If
%                   grid is an array, it is the number of nodes of the
%                   first irregular grid.
%   Method          Similarity functions: ZNCC or COSXCORR
%   Seeds           NX2 array. Seeds contains the coordinates of the nodes
%                   that must be added to the first grid.
%   printInfo       Logical value. If true, some information on the
%                   processing are displayed during the run
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

%Default values
defaultSubPixel = true;
defaultOverSmpl = 10;
defaultMaxBand = [-10 10 -10 10];
defaultTolerance = 2;
defaultMaxScale = [];
defaultMethod = 'cosxcorr';
defaultNeigh = 4;
defaultSeeds = [];
defaultPrintInfo = false;

Opt = inputParser;
addRequired(Opt,'MasterImage',@isnumeric);
addRequired(Opt,'SlaveImage',@isnumeric);
addRequired(Opt,'tileSz',@isnumeric);
addRequired(Opt,'grid',@isnumeric);
addParameter(Opt,'SubPixel',defaultSubPixel,@islogical);
addParameter(Opt,'OverSmpl',defaultOverSmpl,@isnumeric);
addParameter(Opt,'Tolerance',defaultTolerance,@isnumeric);
addParameter(Opt,'MaxScale',defaultMaxScale,@isnumeric);
addParameter(Opt,'MaxBand',defaultMaxBand,@isnumeric);
addParameter(Opt,'Method',defaultMethod,@ischar);
addParameter(Opt,'Neigh',defaultNeigh,@isnumeric);
addParameter(Opt,'Seeds',defaultSeeds,@isnumeric);
addParameter(Opt,'printInfo',defaultPrintInfo,@islogical);

parse(Opt,MasterImage,SlaveImage,tileSz,grid,varargin{:});
SubPixel = Opt.Results.SubPixel;
os = Opt.Results.OverSmpl;
tolerance = Opt.Results.Tolerance;
maxScale = Opt.Results.MaxScale;
maxband = Opt.Results.MaxBand;
Method = Opt.Results.Method;
neigh = Opt.Results.Neigh;
seeds = Opt.Results.Seeds;
printInfo = Opt.Results.printInfo;


%Check inputs
[regularGrid,Pool]=CheckInputCorrectness();


%for simplicity, take half the tile size
tileSz=floor(tileSz/2);

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
        centroidNumber=centroidNumber*2;
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
            if all(isnan(refTile(:))) || all(isnan(searchTile(:))) ...
                    || numel(unique(refTile(:)))==1 || numel(unique(searchTile(:)))==1
                dx(jj)=nan;
                dy(jj)=nan;
                ncc(jj)=nan;
                void(jj)=1;
            else
%====================  MATCHING  ================================
                t=tic;
                %calculate the similarity function
                [dx(jj),dy(jj),DCC] = matching...
                    (refTile,searchTile,[cm,cp,rm,rp],...
                    'subpixel',SubPixel,'oversmpl',os,...
                    'method',Method);
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


%--------------------------------------------------------------------------
    function [regularGrid, Pool]=CheckInputCorrectness
        

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
        %=======
        
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
        if any(size(MasterImage)~=size(SlaveImage))
            error('Master and slave images must have the same size')
        end
        %check whether seeds is correctly defined
        if ~isempty(seeds) && ...
                (size(seeds,2)~=2 || all(seeds<1,'all') || all(mod(seeds,1)~=0))
            error('seeds must be a NX2 array of positive integers')
        end
        %check tileSz
        if numel(tileSz)~=1 || tileSz<1 || mod(tileSz,1)~=0
            error('tileSz must a positive integer')
        end
        %check whether neigh is positive
        if neigh<=0 ||  mod(neigh,1)~=0
            error('neigh must be a positive integer')
        end
        %check grid
        if all(mod(grid,1)~=0,'all') || any(grid(:)<=0)
            error('grid must be a 1x1 or Nx2 array of positive ingers')
        end
        %check whether maxScale is correctly defined
        if ~isempty(maxScale) && (numel(maxScale)~=1 || mod(maxScale,1)~=0)
            error('maxScale must be a positive integer')
        elseif isempty(maxScale) && numel(grid)==1
            maxScale = grid;
        elseif isempty(maxScale) && numel(grid)>1
            maxScale = size(grid,1);
        end
        %check maxband
        if numel(maxband)~=4 || any(mod(maxband(:),1)~=0)
            error('maxband must be a 1x4 array of positive integers')
        end
        %check tolerance
        if numel(tolerance)~=1 || tolerance<0 || mod(tolerance,1)~=0
            error('tolerance must a positive integer')
        end
        %check whether it is set a regular or irregular grid
        if numel(grid)==1
            regularGrid=true;
            %check whether maxScale is correctly defined
            if isempty(maxScale)
                maxScale = grid;
            elseif maxScale<grid
                error('maxScale cannot be lower than grid')
            end
        elseif size(grid,2)==2 
            regularGrid=false;
            %check whether the Statistics Toolbox is installed for using
            %kmedoids
            if ~license('test','Statistics_Toolbox')
                error('The Statistics and Machine Learning Toolbox is required for irregular nodes')
            end

            %check whether maxScale is correctly defined
            if isempty(maxScale)
                maxScale = size(grid,1);
            elseif maxScale>size(grid,1)
                error('maxScale must be lower than grid')
            end
        else
            error('nodes must be a number or a NX2 array')
        end
        %check whether the oversampling option is in correct form
        if os<1 || mod(os,1)~=0
            error('OverSampling must be a positive integer')
        end
    end
%--------------------------------------------------------------------------

end


