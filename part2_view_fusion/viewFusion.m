%%
clear;clc;

nx = 1324;
ny = 1985;
nz = 492;

% read the combined segmentation labels from per view for a rough
% impression about current results
parfor i=1:nz
    i
    img = imread(['combine/slice_' num2str(i-1,'%05d') '.tif']);
    Images(:,:,i)=img;
end

figure; sliceViewer(Images)

%% Read probability maps
% xz
parfor i=1:ny
    i
    slice = imread(['prob/xz/slice_' num2str(i-1,'%05d') '.tif']);
    slice = imresize(slice, [nz nx]);
    StackXY(:,:,i) = slice;
end
% xy
parfor i=1:nx
    i
    slice = imread(['prob/xy/slice_' num2str(i-1,'%05d') '.tif']);
    slice = imresize(slice, [nz ny]);
    StackXZ(:,:,i) = slice;
end
% yz
parfor i=1:nz
    i
    slice = imread(['prob/yz/slice_' num2str(i-1,'%05d') '.tif']);
    slice = imresize(slice, [nx ny]);
    StackYZ(:,:,i) = slice;
end

%% Kmeans clustering for pixels and component connecting to obtain the particle labels
StackProb = uint16(permute(StackXY, [2 3 1]) + permute(StackXZ, [3 2 1]) + StackYZ); % fusion all possibilities

StackProb2=imresize3(StackProb,0.25); % resize the stack to save memory
% StackProb2 = StackProb; % original stack size

Clusters = kmeans([StackProb2(:)],5); % pixel clustering based on the fused probability map
C = reshape(Clusters, size(StackProb2));
Label = bwlabeln(C-1); % connected components to generate the particle labels, background set to be 0
figure; sliceViewer(Label)
colormap('jet')

%% (optional) Remove small components
classes = unique(Label);
Label2 = zeros(size(Label));
parfor ci=2:numel(classes)
   ci
    BW = Label==classes(ci);
    if sum(BW(:))>2000
        Label2 = Label2 + BW*ci;
    end
end
figure; sliceViewer(Label2)
colormap('jet')
%% (optional) Linking
% The labels at different slices can be further linked by hungarian algorithm. 
% See simpletracker.m and hungarianlinker.m for details
%% (optional) Further refinement
% The boundaries of the results from fused probability map may not be
% perfect in some cases. It is suggested to have post-processing steps by combining the
% [Images] or that from single view data, such as simple thresholding within the identified objects.
%% (optional) Repeatedly network detection
% Alternatively, it may be necessary to go through the network detection repeatedly to
% better refine the identification results. One feasible approach is to
% feed the combined [Images] and pseudo labels into the network to train better weights.
% It showed improved performance.

%%
parfor i=1:size(Label,3)
    i
    imwrite((uint16(Label(:,:,i))), ['Seg/slice_' num2str(i-1, '%05d') '.tif']);
end