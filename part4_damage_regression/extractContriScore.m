
%%%%%%%%%%%%%%%%%%% extracting and reordering the results
% feature names are {'Z','Roll','Pitch','Yaw','Edensity','Homogeneity','SurfInner','SurfOuter','Volume','Elongation','Sphericity','Vsratio','RoughInner','RoughOuter','Contact','PDensity','DisNearest','locVolume','locOri','locElongation','locSphericity'}
%
%% 1-results from random seeding
clear all
num_repeat = 10;

for i=1:num_repeat
    load(['data/overall_result_seed' num2str(i-1) '.mat']);
    index = [1 2 3 4 6 5 12 11  10 14 13 8 7 9 16 20 15 19 17 18 21]; % reordering for better illustration
    fn=featurenames(index);
    overall10cyc(:,i) = cyc10(index)'; overall50cyc(:,i) = cyc50(index)';
end
overall10cyc = overall10cyc';
overall50cyc = overall50cyc';

overall10cyc = overall10cyc(:);
overall50cyc = overall50cyc(:);

figure; plot(overall10cyc); hold on; plot(overall50cyc);

%% 2-results from data-subsampling
clear overall10cyc overall50cyc
for i=1:num_repeat
    load(['data/overall_result_seed_subset' num2str(i-1) '.mat']);
    index = [1 2 3 4 6 5 12 11  10 14 13 8 7 9 16 20 15 19 17 18 21];  % reordering for better illustration
    fn=featurenames(index);
    overall10cyc(:,i) = cyc10(index)'; overall50cyc(:,i) = cyc50(index)';
end
overall10cyc = overall10cyc';
overall50cyc = overall50cyc';

overall10cyc = overall10cyc(:);
overall50cyc = overall50cyc(:);
figure; plot(overall10cyc); hold on; plot(overall50cyc)

%% difference between two cycles
diff = overall50cyc-overall10cyc;
clear A B;
k=1;
for i=1:num_repeat:numel(overall10cyc)
    tmp = diff(i:i+9,:);
    A(k) = mean(tmp(:));
    B(k) = std(tmp(:,1));
    k=k+1;
end
A=A';
B=B';

