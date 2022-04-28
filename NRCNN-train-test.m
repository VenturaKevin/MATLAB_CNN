%%formatting 5g DATA for ML 

SNR = [-5 0 5 10 20 30];

mods = {'QPSK';'QAM16';'QAM64';'QAM256'}';
modcat = categorical(mods);

%onehot values for each mod
one_hotm = onehotencode(modcat,1);
%array to store mod for each sample
one_hot = zeros(521,4);

%IQ data array
X = zeros(521,2,128);

qpsk_neg5 = "QPSK_*_neg5snr_TM.mat";
qpsk_0 = "QPSK_*_0snr_TM.mat";
qpsk_5 = "QPSK_*_5snr_TM.mat";
qpsk_10 = "QPSK_*_10snr_TM.mat";
qpsk_20 = "QPSK_*_20snr_TM.mat";
qpsk_30 = "QPSK_*_30snr_TM.mat";

QAM16_neg5 = "sixteenQAM_*_neg5snr_TM.mat";
QAM16_0 = "sixteenQAM_*_0snr_TM.mat";
QAM16_5 = "sixteenQAM_*_5snr_TM.mat";
QAM16_10 = "sixteenQAM_*_10snr_TM.mat";
QAM16_20 = "sixteenQAM_*_20snr_TM.mat";
QAM16_30 = "sixteenQAM_*_30snr_TM.mat";

QAM64_neg5 = "sixtyfourQAM_*_neg5snr_TM.mat";
QAM64_0 = "sixtyfourQAM_*_0snr_TM.mat";
QAM64_5 = "sixtyfourQAM_*_5snr_TM.mat";
QAM64_10 = "sixtyfourQAM_*_10snr_TM.mat";
QAM64_20 = "sixtyfourQAM_*_20snr_TM.mat";
QAM64_30 = "sixtyfourQAM_*_30snr_TM.mat";

QAM256_neg5 = "twofivesixQAM_*_neg5snr_TM.mat";
QAM256_0 = "twofivesixQAM_*_0snr_TM.mat";
QAM256_5 = "twofivesixQAM_*_5snr_TM.mat";
QAM256_10 = "twofivesixQAM_*_10snr_TM.mat";
QAM256_20 = "twofivesixQAM_*_20snr_TM.mat";
QAM256_30 = "twofivesixQAM_*_30snr_TM.mat";

modfiles = [qpsk_neg5 qpsk_0 qpsk_5 qpsk_10 qpsk_20 qpsk_30 QAM16_neg5 QAM16_0 QAM16_5 QAM16_10 QAM16_20 QAM16_30 QAM64_neg5 QAM64_0 QAM64_5 QAM64_10 QAM64_20 QAM64_30 QAM256_neg5 QAM256_5 QAM256_10 QAM256_20 QAM256_30];

j = 1;
for snrval = 1:23
    files = dir(modfiles(snrval));
    for file = 1:length(files)
        
        %try and assign snr vals to each vector
        %make a seperate vector and loop like one hot for snr
        if snrval <= 6
            one_hot(j,:) = one_hotm(1,:);
        elseif snrval > 6 && snrval <= 12
            one_hot(j,:) = one_hotm(2,:);
        elseif snrval > 12 && snrval <= 18
            one_hot(j,:) = one_hotm(3,:);
        elseif snrval > 18 && snrval <= 23
            one_hot(j,:) = one_hotm(4,:);
        end
        data1 = load(files(file).name);
        newStr = erase(files(file).name, '.mat');
        data22 = data1.(newStr);
        data32 = data22.Data;
        % for loop for 128 values/ split into I and Q
  
        for i = 1:128
        
            data42 = data32(i);
            % save into I array 
            X(j,1,i) = real(data42);
            % save into Q array
            X(j,2,i) = imag(data42);

        end
        j = j + 1;   
    end  
end

%save('NRdataset.mat','one_hot','mods','SNR','X','-v7','-nocompression')

%%
%re shaping data into input size of CNN
% 1st three dimensions, frame size. Last dimension how many frames
Xtrain = permute(X,[4,3,2,1]);
[v,p] = max(one_hot');

Ytrain = categorical(p);
for i = 1:length(mods)
    Ytrain(p==i) = mods(i);
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

%save('DATA.mat','Xtrain','Ytrain','Yval','Xval', 'SNR','-v7','-nocompression')

%%

layers = customCNN([1,128,2],4);
save('DataNet.mat', 'layers')
%analyzeNetwork(layers);

%%
mbs = 64;
valfreq = floor(numel(Ytrain)/mbs);

options = trainingOptions("sgdm",...
    "ValidationData",{Xval,Yval},...
    "ValidationFrequency",valfreq,...
    "Verbose",false,...
    "Plots","training-progress",...
    "MiniBatchSize",mbs);

[net,info] = trainNetwork(Xtrain,Ytrain,layers,options);

%%
testprediction = classify(net,Xval);
conf = confusionchart(Yval,testprediction);
conf.RowSummary = 'row-normalized';


