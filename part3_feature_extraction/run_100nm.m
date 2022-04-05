pixelsize = 100;

load('data/100nm_seg_good');
load('data/100nm_data_good');

%% remove the boundary pixels
Seg_good = single(Seg_good(31:end-30, 31:end-30, 31:end-30));
Data_resize= single(Data_resize(31:end-30, 31:end-30, 31:end-30));

%% extract all features and save the results
allfeatures = getAllFeatures(Data_resize, Seg_good, pixelsize);

save results_features_100 allfeatures;