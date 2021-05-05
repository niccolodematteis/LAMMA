
clear
close all

%load images
masterImage=imread('glacier1.jpg');
slaveImage=imread('glacier2.jpg');

%set parameters
tileSz=128;
nodeDist=32;
os=10;
SubPix=true;
band=[-30 30 -10 70];
maxscale=256;
tol=2;
method='cosxcorr';
neigh=4;

%launch LAMMA
tic
[DX,DY,NCC,nodes,calcNumber,BDlimits] = LAMMA...
    (masterImage,slaveImage,tileSz,...
    nodeDist,'oversmpl',os,'subpixel',SubPix,...
    'maxband',band,'maxscale',maxscale,...
    'tolerance',tol,'method',method,'neigh',neigh,...
    'printinfo',true);
toc

disp(' ')
disp('----------------------------------------------')
NPlamma = sum(calcNumber);
NPcan = numel([DX{:}])*( (abs(band(2)-band(1))+1)*(abs(band(4)-band(3))+1) ); 
disp(['LAMMA complexity = ',num2str(NPlamma)])
disp(['Canonical complexity = ',num2str(NPcan)]);
disp([num2str(round(NPcan/NPlamma)),' times less calculi'])
disp('----------------------------------------------')

%display results
ncol = 256;
cmap = jet(ncol);
thr=1;
c=vertcat(nodes{:});
x=[DX{:}];
y=[DY{:}];
v=sqrt(x.^2+y.^2);
v(v<thr)=nan;
x=x./v;
y=y./v;
bin = logspace(-1,2,ncol);
bin = [bin,Inf];

figure
imshow(masterImage)
hold on
for ii=1:numel(bin)-1
    pun=v>=bin(ii) & v<bin(ii+1);
    quiver(c(pun,1),c(pun,2),50*x(pun)',50*y(pun)',0,'color',cmap(ii,:))
end
hold off
axis equal tight
colormap(cmap)
cl=colorbar;
cl.Label.String='px';
caxis([0.1 100])
set(gca,'colorscale','log')
title('Displacement field')

