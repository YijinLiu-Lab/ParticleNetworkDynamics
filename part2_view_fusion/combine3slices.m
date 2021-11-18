clear; clc;
% xy slice
for i=0:1984
    i
    tmp = double(imread(['xy_mask\slice_' num2str(i, '%04d') '.tif']));
    MaskXY(:,:,i+1) = tmp;
end
% xz slice
for i=1:1324
    i
    tmp = double(imread(['z_mask\xzslice_' num2str(i, '%04d') '.tif']));
    MaskXZ(:,:,i+1) = tmp;
end
% xy slice
for i=1:492
    i
    tmp = double(imread(['yz_mask\yzslice_' num2str(i, '%04d') '.tif']));
    MaskYZ(:,:,i+1) = tmp;
end

%%
MaskXZ2 = MaskXZ(:,:, 1:1324);
MaskYZ2 = MaskYZ(:,:,1:492);

MaskXZ2 = permute(MaskXZ2, [1 3 2]);
MaskYZ2 = permute(MaskYZ2, [3 1 2]);

MASK = MaskXY + MaskXZ2 + MaskYZ2;

%%
I = MaskXY;
gmag = imgradient(I);
imshow(gmag(:,:,1),[])
title('Gradient Magnitude')





