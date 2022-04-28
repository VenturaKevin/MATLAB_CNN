%re shaping data into input size of CNN
% 1st three dimensions, frame size. Last dimension how many frames
Xtrain = permute(data,[4,2,3,1]);
[v,p] = max(onehot');

Ytrain = categorical(p);
for i = 1:length(mods)
    Ytrain(p==i) = mods(i,:);
end
%trick
Ytrain = string(Ytrain);
Ytrain = categorical(Ytrain);

% this split dataset into a validation set. 20% of dataset. 
% can re use for test split
idx = randperm(size(Xtrain,4),round(size(Xtrain,4)*.20));
Xval = Xtrain(:,:,:,idx);
Xtrain(:,:,:,idx) = [];
Yval = Ytrain(idx);
Ytrain(idx) = [];

save('DATA1.mat','Xtrain','Ytrain','Yval','Xval', 'snr','-v7','-nocompression')

%%

layers = customCNN1([1,128,2],11);
save('DataNet1.mat', 'layers')
analyzeNetwork(layers);

%%
mbs = 64;
valfreq = floor(numel(Ytrain)/mbs);
ep = 4;

options = trainingOptions("sgdm",...
    "ValidationData",{Xval,Yval},...
    "ValidationFrequency",valfreq,...
    "Verbose",false,...
    "Plots","training-progress",...
    "MiniBatchSize",mbs,...
    "MaxEpochs", ep);

[net,info] = trainNetwork(Xtrain,Ytrain,layers,options);

%%
testprediction = classify(net,Xval);
conf = confusionchart(Yval,testprediction);
conf.RowSummary = 'row-normalized';
