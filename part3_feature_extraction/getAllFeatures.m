function allfeatures = getAllFeatures(Data, Seg, pixelsize)

[nx,ny,nz] = size(Seg);
volume = zeros(1,max(Seg(:)));
p_x = zeros(1,max(Seg(:)));
p_y = zeros(1,max(Seg(:)));
p_z = zeros(1,max(Seg(:)));
Mean = zeros(1,max(Seg(:)));
Std = zeros(1,max(Seg(:)));
elongation = zeros(1,max(Seg(:)));

damage = zeros(1,max(Seg(:)));
V3S2 = zeros(1,max(Seg(:)));
PALmax = zeros(1,max(Seg(:)));
contact = zeros(1,max(Seg(:)));

pdensity = zeros(1,max(Seg(:)));
local_porosity = zeros(1,max(Seg(:)));
inner_surf = zeros(1,max(Seg(:)));
outer_surf = zeros(1,max(Seg(:)));
inner_surf_rough = zeros(1,max(Seg(:)));
outer_surf_rough = zeros(1,max(Seg(:)));

% cube size
radiusCube = round(10000/pixelsize);

mask = single(zeros(size(Seg)));
mask(radiusCube+1:size(Seg,1)-radiusCube , radiusCube+1:size(Seg,2)-radiusCube , radiusCube+1:size(Seg,3)-radiusCube) = 1;

A = unique(Seg(:));
A = A(2:end);

%% Particle own features
disp('begin extracting particle features...');
tic;
for i = 1:length(A)
    disp(['particle ' num2str(i) '/' num2str(length(A))]);
    n = A(i);
    Seg_P = double(Seg == n);
    
    stats = regionprops3(Seg_P,'all');
    p_x(i) = stats.Centroid(2);       % x
    p_y(i) = stats.Centroid(1);       % y
    p_z(i) = stats.Centroid(3);       % z
    
    cord(i,:) = [round(p_x(i)),round(p_y(i)),round(p_z(i))];
    
     %% store information
    allfeatures(i).OriX = stats.Orientation(1);
    allfeatures(i).OriY = stats.Orientation(2);
    allfeatures(i).OriZ = stats.Orientation(3);
    
    %% store information
    allfeatures(i).pX = p_x(i);
    allfeatures(i).pY = p_y(i);
    allfeatures(i).pZ = p_z(i);
    
    PALmax(i) = max([stats.PrincipalAxisLength(1),stats.PrincipalAxisLength(2),stats.PrincipalAxisLength(3)]);
    PALmin = min([stats.PrincipalAxisLength(1),stats.PrincipalAxisLength(2),stats.PrincipalAxisLength(3)]);
    elongation(i) = PALmax(i) / PALmin;
    %% store information
    allfeatures(i).Elongation = elongation(i);
    
    y_l = round(stats.BoundingBox(1));
    y_r = round(stats.BoundingBox(1))+stats.BoundingBox(4);
    x_l = round(stats.BoundingBox(2));
    x_r = round(stats.BoundingBox(2))+stats.BoundingBox(5);
    z_l = round(stats.BoundingBox(3));
    z_r = round(stats.BoundingBox(3))+stats.BoundingBox(6);
    Seg_P = Seg_P(x_l:x_r , y_l:y_r , z_l:z_r);             % binary particle with cracks
%     Data_P = Data;
    Data_P_all = Data(x_l:x_r , y_l:y_r , z_l:z_r);           % original data without cracks
    
    se = strel('sphere',10);
    Seg_P_fill = imclose(Seg_P,se);
    Seg_P_fill = imfill(Seg_P_fill);
    volume(i) = sum(Seg_P_fill(:));
    Data_P_all = Data_P_all .* Seg_P_fill;
    Mean(i) = mean(Data_P_all(Data_P_all~=0));     % mean
    Std(i) = std(Data_P_all(Data_P_all~=0));       % std
    
    %% store information
    allfeatures(i).Volume = volume(i);
    allfeatures(i).EDensity = Mean(i);
    allfeatures(i).Homogeneity = Std(i);
    
    
    outer_surf_tmp = bwperim(Seg_P_fill);
    outer_surf(i) = sum(outer_surf_tmp(:));
    outer_surf_filt = medfilt3(Seg_P_fill);
    outer_surf_filt = bwperim(outer_surf_filt);
    outer_surf_filt = sum(outer_surf_filt(:));
    outer_surf_rough(i) = outer_surf_filt / outer_surf(i);
    
    total_surf_P = Seg_P;
    total_surf_tmp = bwperim(total_surf_P);
    total_surf(i) = sum(total_surf_tmp(:));
    
    total_surf_filt = medfilt3(total_surf_P);
    total_surf_filt = bwperim(total_surf_filt);
    total_surf_filt =  sum(total_surf_filt(:));
    
    inner_surf_filt = total_surf_filt - outer_surf_filt;
    inner_surf(i) = total_surf(i) - outer_surf(i);
    inner_surf_rough(i) = inner_surf_filt / inner_surf(i);
    
    V3S2(i) = (volume(i))^(1/3) / (total_surf(i))^(1/2);
    
    %% store information
    allfeatures(i).SurfOuter = outer_surf(i);
    allfeatures(i).SurfInner = inner_surf(i);
    allfeatures(i).RoughInner = inner_surf_rough(i);
    allfeatures(i).RoughOuter = outer_surf_rough(i);
    allfeatures(i).VSratio = V3S2(i);
    allfeatures(i).Sphericity = 36*pi*(volume(i)^2)/(total_surf(i)^3);
    
    
    V_tmp = Seg_P_fill - Seg_P;
    V_hole_crack = length(V_tmp(V_tmp==1));
    damage(i) = V_hole_crack / volume(i);
    %% store information
    allfeatures(i).Damage = damage(i);
  
    lX = max(1, x_l-radiusCube);
    rX = min(nx, x_r+radiusCube);
    
    lY = max(1, y_l-radiusCube);
    rY = min(ny, y_r+radiusCube);
    
    lZ = max(1, z_l-radiusCube);
    rZ = min(nz, z_r+radiusCube);
    
    particle_around = Seg(lX:rX , lY:rY, lZ:rZ);
    mask_P = mask;
    particle_V = mask_P(lX:rX , lY:rY, lZ:rZ);
    particle_V = sum(particle_V(:));
        
    particle_around_only = particle_around;
    particle_around_only(particle_around_only ~= n) = 0;
    SE = strel('sphere',10);
    particle_around_only_fill = imclose(particle_around_only,SE);
    particle_around_only_fill = imfill(particle_around_only_fill);
    particle_around_only_dilate = imdilate(particle_around_only_fill,SE);
    particle_around_only_ring = particle_around_only_dilate - particle_around_only_fill;

    
    particle_around_only_ring(particle_around_only_ring ~= 0) = 1;
    particle_around_only_times = particle_around_only_ring .* particle_around;
    particle_around_only_times(particle_around_only_times ~= 0) = 1;
    particle_contact(i) = sum(particle_around_only_times(:)) / sum(particle_around_only_ring(:));
    
    %% store information
    allfeatures(i).Contact = particle_contact(i);
    
    particle_around_V = length(particle_around(particle_around~=i)) - length(particle_around(particle_around==1)) - length(particle_around(particle_around==0));
    density(i) = particle_around_V / particle_V;
    local_porosity(i) = (particle_V - length(particle_around(particle_around~=0))) / particle_V;
    
    allfeatures(i).PDensity = 1-local_porosity(i);
    
end

toc

%% Neighbor features
dis_tmp = pdist(cord,'euclidean');
distance = squareform(dis_tmp);

mean_neighbour_P = zeros(1,max(Seg(:)));
std_neighbour_P = zeros(1,max(Seg(:)));
dis_nearest_P = zeros(1,max(Seg(:)));

numNeighbor = 6;

disp('begin extracting neighbor features...');
tic;

for ii = 1:length(A)
    disp(['particle ' num2str(ii) '/' num2str(length(A))]);
    
    A = distance(:,ii);
    B = find(A > 0 & A < sqrt(2)*radiusCube);
    
    tmp = A(A ~= 0);
    [tmp2, index] = sort(tmp);
    
    dis_nearest_P(ii) = mean(tmp2(1: min(numNeighbor, length(tmp))));
    
%     Edensities = allfeatures(index).Edensity;
%     Homogeneities = allfeatures(index).Homogeneity;
    
    OriXs = allfeatures(index).OriX;
    OriYs = allfeatures(index).OriY;
    OriZs = allfeatures(index).OriZ;
    
    Volumes = allfeatures(index).Volume;
    
    Elongations = allfeatures(index).Elongation;
    
    Sphericities = allfeatures(index).Sphericity;
    
    %% store information
    allfeatures(ii).DisNearest = dis_nearest_P(ii);
    
    % size
    allfeatures(ii).locVolume = std(Volumes);
    allfeatures(ii).locOri = max([std(OriXs) std(OriYs) std(OriZs)]);
    
    allfeatures(ii).locElongation = std(Elongations);
    allfeatures(ii).locSphericity = std(Sphericities);
    
end

toc


end