% 1: 导入数据
clear, close all
excelFilePath = '\Runoff_JINZHOU_filtered.xlsx';
excelData = readtable(excelFilePath);
streamflow = excelData.Q; % Assuming 'Q' is the column for streamflow data
T = 1:numel(streamflow);

PetFilePath = "\PET_Jinzhou_filtered.xlsx";
PetData = readtable(PetFilePath);
PM = PetData.PM_PET; 

MeteoFilePath = "\meteorology_Jinzhou_filtered.xlsx";
MeteoData = readtable(MeteoFilePath);
PRCP = MeteoData.PRCP;

% 提取指定列作为 Meteo 数据 (10列)
Meteo = MeteoData(:, {'TEMP', 'DEWP', 'SLP', 'VISIB', 'WDSP', 'MXSPD', 'MAX', 'MIN', 'SNDP', 'FRSHTT'});
Meteo = table2array(Meteo); % Convert table to array

% %% 
% % 2: 可视化数据
% figure;
% hold on;
% plot(PRCP, 'b-', 'DisplayName', 'PRCP');
% plot(PM, 'r-', 'DisplayName', 'PM');
% hold off;
% title('PRCP and PM Plot');
% xlabel('Time');
% ylabel('Value');
% legend('show');
%% 

% 3: 数据正规化 (使用所有13个变量)
RawData = [streamflow, PM, PRCP, Meteo];
[nData, mu, sigma] = zscore(RawData);

%% 
% 卷积
zscoreMeteo = nData(:, 4:end);  % 提取Meteo数据
numSamples = size(zscoreMeteo, 1);  % 获取样本数量
zscoreMeteo = reshape(zscoreMeteo', [10, numSamples, 1]);  % 转换为 [特征数, 样本数, 1]
% 定义卷积核
kernel = ones(1, 10) / 10;  % 你可以根据需要定义不同的卷积核
% 对每一列进行卷积
MeteoConv = zeros(1, numSamples);  % 初始化结果变量
for i = 1:numSamples
    MeteoConv(i) = conv(zscoreMeteo(:, i), kernel, 'valid');  % 'valid' 选项确保输出维度匹配
end
alpha = -0.01; % 设置斜率
MeteoConv = leakyReLU(MeteoConv, alpha);
% 结果现在是一个 1×12053 的变量
% figure
% plot(MeteoConv)
nDataConv = [nData(:, 1:3), MeteoConv(:)]; 

% 4: 数据分割 (90%训练, 10%验证)
NumTrain = floor(0.9 * size(nDataConv, 1));
TrainData = nDataConv(1:NumTrain, :);
TestData = nDataConv(NumTrain+1:end, :);

%% 

% 5: LSTM网络的构建
layers = [
    sequenceInputLayer(4)                              % 4个输入特征 (streamflow, PM, PRCP, Meteo)
    lstmLayer(200)                                     % LSTM层
    fullyConnectedLayer(1)                             % 输出层
    regressionLayer                                    % 回归层
];


%6：设置学习选项
options = trainingOptions('sgdm', ...
    'MaxEpochs',60, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.001, ...
    'LearnRateSchedule','piecewise', ...
    'GradientThresholdMethod','global-l2norm',...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',1, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment', 'gpu');

% 7: 执行学习
% 调整 XTrain 的维度为 [4, 时间步数, 1] (特征数, 时间步数, 序列数)
XTrain = permute(TrainData(1:end-1, :), [2, 1, 3]); % [4, 训练步数, 1]
% 目标值 (streamflow的时间序列)
YTrain = TrainData(2:end, 1)';

doTrain = true;

if doTrain
    net = trainNetwork(XTrain, YTrain, layers, options);
    save('TrainedLSTMNetwork', 'net');
    net0 = net;
else
    PreTrainedStr = load("TrainedLSTMNetwork");
    net = PreTrainedStr.net;
    net0 = net;
end
%% 

% 8: 递归预测
NumTest = size(TestData, 1) - 1;
YPred = zeros(1, NumTest);

for n = 1:NumTest
    XTest_Meteo = permute(TestData(n, 4:end), [2, 3, 1]);   % 测试Meteo数据
    XTest_Other = permute(TestData(n, 1:3), [2, 3, 1]);    % 测试重要变量 (streamflow, PM, PRCP)
    XTest = [XTest_Other; XTest_Meteo];                    % 合并为LSTM输入

    [net, YPred(n)] = predictAndUpdateState(net, XTest);
end

% 计算 RMSE
RMSE = sqrt(mean((YPred - TestData(2:end, 1)').^2));
disp(['RMSE = ', num2str(RMSE)]);

% 9: 反归一化预测值 (streamflow)
rYPred = sigma(1) .* YPred + mu(1);

% 10: 可视化预测结果
figure;
axH = axes;
pH1 = plot(axH, T, RawData(:, 1), 'DisplayName', 'Observation');
hold on;
pH2 = plot(axH, T(NumTrain+1:end), [RawData(NumTrain+1, 1); rYPred'], ".-", 'DisplayName', 'Forecast');
fH = fill(axH, T([1, NumTrain, NumTrain, 1]), [repmat(axH.YLim(1), 1, 2), repmat(axH.YLim(2), 1, 2)], [0.9, 0.9, 0.9], 'EdgeColor', 'none', 'DisplayName', 'Training Period');
uistack(fH, "bottom");
hold off;
xlabel("Time (Days)");
ylabel("Streamflow");
title("Forecast Streamflow using LSTM");
legend([pH1, pH2, fH], "Location", "best");
