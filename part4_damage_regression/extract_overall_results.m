
%%%%%%%%%%%%%%%%%%% reordering

for i=1:10
    load(['overall_result_seed' num2str(i-1) '.mat']);
    index = [1 2 3 4 6 5 11 12 10 14 13 8 7 9 16 20 15 19 17 18 21];
    fn=featurenames(index);
    overall(:,i) = cyc10(index)'; overall(:,i+10) = cyc50(index)';
end


%%
clear overall10cyc overall50cyc
for i=1:10
    load(['overall_result_seed' num2str(i-1) '.mat']);
    index = [1 2 3 4 6 5 12 11  10 14 13 8 7 9 16 20 15 19 17 18 21];
    fn=featurenames(index);
    overall10cyc(:,i) = cyc10(index)'; overall50cyc(:,i) = cyc50(index)';
end
overall10cyc = overall10cyc';
overall50cyc = overall50cyc';

overall10cyc = overall10cyc(:);
overall50cyc = overall50cyc(:);

figure; plot(overall10cyc); hold on; plot(overall50cyc)
%%
for i=1:10
    load(['overall_result_seed_subset' num2str(i-1) '.mat']);
    index = [1 2 3 4 6 5 11 12 14 10 13 8 7 9 16 20 15 19 17 18 21];
    fn=featurenames(index);
    overall(:,i) = cyc10(index)'; overall(:,i+10) = cyc50(index)';
end

%%
%%
clear overall10cyc overall50cyc
for i=1:10
    load(['overall_result_seed_subset' num2str(i-1) '.mat']);
    index = [1 2 3 4 6 5 12 11  10 14 13 8 7 9 16 20 15 19 17 18 21];
    fn=featurenames(index);
    overall10cyc(:,i) = cyc10(index)'; overall50cyc(:,i) = cyc50(index)';
end
overall10cyc = overall10cyc';
overall50cyc = overall50cyc';

overall10cyc = overall10cyc(:);
overall50cyc = overall50cyc(:);
figure; plot(overall10cyc); hold on; plot(overall50cyc)

%%
figure;
clear Test Pred;
% fitting accuracy
for i=1:10
    load(['fitting_features_50cyc.csv_' num2str(i-1) '.mat']);
    error(i) = mean(abs((y_test-y_pred)));
    Test(i,:) = y_test;
    Pred(i,:) = y_pred;
    plot(y_test,y_pred,'o'); axis equal;
    hold on;
end
figure; plot(error)

%%
% diff = overall50cyc-overall10cyc;
diff = C;
clear A B;
k=1;
for i=1:10:210
    tmp = diff(i:i+9,:);
    A(k) = mean(tmp(:));
    B(k) = std(tmp(:,1));
    k=k+1;
end
A=A';
B=B';
