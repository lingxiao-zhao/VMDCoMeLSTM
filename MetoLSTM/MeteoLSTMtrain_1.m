% 1：导入数据
clear, close all
excelFilePath = '\Runoff_SHENYANG.xlsx';
excelData = readtable(excelFilePath);
streamflow = excelData.Q; % Assuming 'Q' is the column for streamflow data

% PetFilePath = '\PET_Shenyang_merged_data.xlsx';
% PetData = readtable(PetFilePath);
% PM = PetData.PM_PET;

MeteoFilePath = "\meteorology_Shenyang_merged_data.xlsx";
MeteoData = readtable(MeteoFilePath);
PRCP = MeteoData.PRCP;

% Extract the specified columns for Meteo and convert to array
Meteo = MeteoData(:, {'TEMP', 'DEWP', 'SLP', 'VISIB', 'WDSP', 'MXSPD', 'MAX', 'MIN', 'SNDP', 'FRSHTT'});
Meteo = table2array(Meteo);

% 2：数据拼接
RawData = [streamflow, PRCP, Meteo]; % 13 columns total

%% 3：数据正规化
[nData, mu, sigma] = zscore(RawData); % Normalize data

%% 4：数据分割
NumTrain = floor(0.9 * size(nData, 1));

TrainData = nData(1:NumTrain, :); % First 90%
TestData = nData(NumTrain+1:end, :); % Last 10%

%% 5：LSTM网络构建
% 修改输入层的特征维度为 13，因为现在有 13 个输入特征
layers = [ ...
    sequenceInputLayer(12)  % Adjusted to 12 input features
    lstmLayer(200)
    fullyConnectedLayer(1)
    regressionLayer];

%% 6：设置训练选项
options = trainingOptions('adam', ...
    'MaxEpochs',70, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.001, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',1, ...
    'Plots','training-progress');

%% 7：执行学习
% 调整 XTrain 的维度为 [14, 时间步数, 1] (特征数, 时间步数, 序列数)
XTrain = permute(TrainData(1:end-1, :), [2, 1, 3]); % [14, 训练步数, 1]
YTrain = TrainData(2:end, 1)';  % 预测的是流量，变为行向量

doTrain = true;

if doTrain
    net = trainNetwork(XTrain, YTrain, layers, options);  % 训练网络
    save('TrainedLSTMNetwork', 'net');
else
    PreTrainedStr = load("TrainedLSTMNetwork");
    net = PreTrainedStr.net;
end

%% 8: 递归预测
NumTest = size(TestData, 1) - 1;  
YPred = zeros(1, NumTest);

for n = 1:NumTest
    XTest = permute(TestData(n, :), [2, 3, 1]);  % [14, 1, 1] 输入维度
    [net, YPred(n)] = predictAndUpdateState(net, XTest);
end

%% 9: 计算 RMSE
RMSE = sqrt(mean((YPred - TestData(2:end, 1)').^2));  % 流量预测
disp(['RMSE = ', num2str(RMSE)]);

% 逆正规化流量预测
rYPred = sigma(1) .* YPred + mu(1);  % 使用流量的均值和标准差进行反向变换

% 10: 可视化预测结果
figure;
T = 1:numel(streamflow);  % 时间轴
plot(T, RawData(:, 1), 'DisplayName', 'Observation'); % 观察到的流量
hold on;
plot(T(NumTrain+1:end), [RawData(NumTrain+1, 1); rYPred'], '.-', 'DisplayName', 'Forecast'); % 预测流量
hold off;
xlabel('Time (Days)');
ylabel('Streamflow');
title('Forecast Streamflow using LSTM');
legend('show');
