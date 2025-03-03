function             [NetoutTest, Training_time,Testing_time,train_RMSE,train_MAPE,test_RMSE,test_MAPE] = bls_train_amwcbls(train_x,train_y,test_x,test_y,WF,WeightEnhan,s,NumFea,NumWin)

tic
H1 = [train_x,  0.1 * ones(size(train_x,1),1)];
y=zeros(size(train_x,1),NumWin*NumFea);
for i=1:NumWin
    WeightFea=WF{i};
    A1 = H1 * WeightFea;A1 = mapminmax(A1);%Map matrix row minimum and maximum values to [-1 1].
    clear WeightFea;
    WeightFeaSparse  = sparse_amwcbls(A1,H1,1e-3,50)';
    WFSparse{i}=WeightFeaSparse;
    
    T1 = H1 * WeightFeaSparse;
    [T1,ps1]  =  mapminmax(T1',0,1);
    T1 = T1';
    
    ps(i)=ps1;
    y(:,NumFea*(i-1)+1:NumFea*i)=T1;
end

clear H1;
clear T1;
H2 = [y,  0.1 * ones(size(y,1),1)];
T2 = H2 * WeightEnhan;
epsilon=2^3;
lamd = 2^-30; 
gamma=lamd*epsilon^2;
T2 = tansig(T2); %T3代表A
T3=[y T2];


clear H2;
clear T2;
nc=size(train_y,2);% 权重矩阵W的列数
nr=size(T3,2); %权重矩阵W的行数
WeightTop = randn(nr,1);%初始化权重矩阵W
n=size(train_y,1);
Q=zeros(n,1);
converged=false;
t=0;
while  ~converged
t=t+1;
y222=T3*WeightTop;
r=abs(y222-train_y);
beta=2./(1+exp(r));
for i=1:n
    Q(i,1)=exp(-(beta(i,1)*(norm(T3(i,1)*WeightTop-train_y(i,1),2)^2/(2*epsilon^2))));
    i=i+1;
end
S=diag(Q);
WeightTop_new=(T3' * S * T3 +gamma * eye(size(T3',1)))\(T3' *S* train_y);%得到W
kesai1 = norm(WeightTop_new-WeightTop,2);

if kesai1 > 0.05 && t<=100 %不动点法收敛条件
    converged = false;
    WeightTop = WeightTop_new;
else
    converged = true;
    WeightTop = WeightTop_new;
end
end

clear i j;

    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Training_time = toc;
disp('Training has been finished!');
disp(['The Total Training Time is : ', num2str(Training_time), ' seconds' ]);
NetoutTrain = T3 * WeightTop;
clear T3 nr nc;

RMSE = sqrt(sum((NetoutTrain-train_y).^2)/size(train_y,1));
MAPE = sum(abs(NetoutTrain-train_y))/mean(train_y)/size(train_y,1);

train_RMSE = RMSE; 
train_MAPE = MAPE;
fprintf(1, 'Training RMSE is : %e, Training MAPE is: %e\n', RMSE, MAPE);
tic;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
HH1 = [test_x .1 * ones(size(test_x,1),1)];
yy1=zeros(size(test_x,1),NumWin*NumFea);
for i=1:NumWin
    WeightFeaSparse=WFSparse{i};ps1=ps(i);
    TT1 = HH1 * WeightFeaSparse;
    TT1  =  mapminmax('apply',TT1',ps1)';
    
    clear WeightFeaSparse; clear ps1;
    yy1(:,NumFea*(i-1)+1:NumFea*i)=TT1;
end
clear TT1;clear HH1;
HH2 = [yy1 .1 * ones(size(yy1,1),1)];
% TT2 = tansig(HH2 * b2 * l2);
TT2 = tansig(HH2 * WeightEnhan);
TT3=[yy1 TT2];
clear HH2;clear b2;clear TT2 i j aRMSE aRRMSE;
NetoutTest1=[];
NetoutTest = TT3 * WeightTop;
NetoutTest1=[NetoutTest1;test_x NetoutTest];
%xlswrite('.xlsx',NetoutTest1);

clear TT3;

%%%%%%以上是训练过程
%%%%%%测试数据预测过程
%% Calculate the testing accuracy   
Testing_time = toc;
disp('Testing has been finished!');
disp(['The Total Testing Time is : ', num2str(Testing_time), ' seconds' ]);
nr_test = size(test_y,1);

RMSE = sqrt(sum((NetoutTest-test_y).^2)/nr_test);
MAPE = mean(sum(abs(NetoutTest-test_y)/test_y));

test_RMSE = RMSE; 
test_MAPE  = MAPE;
fprintf(1, 'Testing RMSE is : %e, Testing MAPE is: %e\n', RMSE, MAPE);

