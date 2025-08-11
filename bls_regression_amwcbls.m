close all;
clear all;
warning off all;
format compact;
load BK.mat; %load .mat

assert(isfloat(train_x), 'train_x must be a float');
assert(isfloat(test_x), 'test_x must be a float');
  
s = .8;              %----s: the shrinkage parameter for enhancement nodes
best = 2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We apply the grid search on the test data set for instance and simplicity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for NumFea=1:20 %searching range for feature nodes  per window in feature layer
    for NumWin=1:20 %searching range for number of windows in feature layer
        for NumEnhan=1:200 %searching range for enhancement nodes
            clc;
            rand('state',1)
            for i=1:NumWin
                WeightFea=2*rand(size(train_x,2)+1,NumFea)-1;
                WF{i}=WeightFea;
            end                                                          %generating weight and bias matrix for each window in feature layer
            %             if NumFea*NumWin>=NumEnhan

            WeightEnhan=2*rand(NumWin*NumFea+1,NumEnhan)-1;
            fprintf(1, 'Fea. No.= %d, Win. No. = %d, Enhan. No. = %d\n', NumFea, NumWin, NumEnhan);
            
            [NetoutTest, Training_time,Testing_time,train_RMSE,train_MAPE,test_RMSE,test_MAPE] = bls_train_amwcbls(train_x,train_y,test_x,test_y,WF,WeightEnhan,s, NumFea,NumWin);
            time =Training_time + Testing_time;
       
            if best > test_RMSE
                best = test_RMSE;
                save('BK1.mat','train_RMSE','train_MAPE','test_RMSE','test_MAPE','NumFea', 'NumWin', 'NumEnhan','time','Training_time','Testing_time','NetoutTest');
            end
            clearvars -except best NumFea NumWin NumEnhan train_x train_y test_x test_y   C s result NetoutTest
        end
    end
    toc
end







