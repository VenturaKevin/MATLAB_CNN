function net = resnetLayers(inputSize, numClasses, options)
% resnetLayers   Create 2-D residual network
%
%   net = resnetLayers(inputSize, numClasses) creates a residual network.
%   inputSize is the input size of the network, specified as a row vector
%   of two or three numbers. numClasses is the number of classes.
%
%   net = resnetLayers(inputSize, numClasses, 'PARAM1', VAL1, 'PARAM2', VAL2, ...)
%   specifies optional parameter name-value arguments for creating the
%   network:
%
%       'InitialFilterSize'   - Height and width of the filters for the
%                               initial convolutional layer. This can be a
%                               scalar, in which case the filters will have
%                               the same height and width, or a vector
%                               [h w] where h specifies the height for the
%                               filters, and w specifies the width. The
%                               default is 7.
%       'InitialNumFilters'   - The number of filters in the initial
%                               convolutional layer. The default is 64.
%       'InitialStride'       - Stride of the initial convolutional layer.
%                               This can be a scalar, in which case the
%                               same value is used for both dimensions, or
%                               it can be a vector [u v] where u is the
%                               vertical stride, and v is the horizontal
%                               stride. The default is 2.
%       'InitialPoolingLayer' - Pooling layer before the first residual
%                               block, specified as one of the following:
%                                - 'max'        - Max pooling layer
%                                - 'average'    - Average pooling layer
%                                - 'none'       - No pooling layer
%                               The default is 'max'.
%       'ResidualBlockType'   - Residual block type, specified as one of
%                               the following:
%                                - 'batchnorm-before-add'
%                                               - Batchnorm layer appears
%                                                 before the addition
%                                                 layer [1].
%                                - 'batchnorm-after-add'
%                                               - Batchnorm layer appears
%                                                 after the addition layer
%                                                 [2].
%                               The default is 'batchnorm-before-add'.
%       'BottleneckType'      - Bottleneck type to use in the block,
%                               specified as one of the following:
%                               - 'none'        - No bottleneck.
%                               - 'downsample-first-conv'    
%                                               - Bottleneck that reduces
%                                                 the amount of computation
%                                                 and is suitable for
%                                                 deeper networks [1].
%                                                 Down-sampling is
%                                                 performed in these blocks
%                                                 by using a stride of 2 in
%                                                 the first convolutional
%                                                 layer in the block.
%                               The default is 'downsample-first-conv'.
%       'StackDepth'          - Number of residual blocks in each stack of
%                               the network, specified as a row vector. For
%                               example, if 'StackDepth' is [3 4 6 3], the
%                               network will have 3 blocks, followed by 4
%                               blocks, then 6 blocks, and finally 3
%                               blocks. The residual blocks in each stack
%                               can have different numbers of filters, as
%                               specified by 'NumFilters'. The default is
%                               [3 4 6 3].
%       'NumFilters'          - Number of filters in the convolutional
%                               layer of each stack. This is a row vector
%                               the same length as 'StackDepth'. The
%                               default is [64 128 256 512]. If
%                               'BottleneckType' is set to
%                               'downsample-first-conv', then this is
%                               before the expansion factor of 4. The
%                               default is [64 128 256 512].
%       'Normalization'       - Data normalization applied when data is
%                               forward propagated through the input layer,
%                               specified as one of the following:
%                                 'zerocenter'  - zero-center normalization
%                                 'zscore'      - z-score normalization
%                               The default is 'zerocenter'.
%
%   Example 1:
%       % Generate resnet-34.
%
%       lgraph = resnetLayers([224 224 3], 1000, 'BottleneckType', 'none');
%
%   Example 2:
%       % Generate a resdiual network suitable for 32-by-32 colour images.
%
%       lgraph = resnetLayers([32 32 3], 10, ...
%           "InitialFilterSize", 3, ...
%           "InitialNumFilters", 16, ...
%           "InitialStride", 1, ...
%           "InitialPoolingLayer", "none", ...
%           "NumFilters", [16 32 64], ...
%           "StackDepth", [4 3 2], ...
%           "ResidualBlockType", "batchnorm-before-add", ...
%           "BottleneckType", "downsample-first-conv");
%
% References
% ----------
% [1] He, Kaiming, Zhang, Xiangyu, Ren, Shaoqing, and Sun, Jian.
%     "Deep Residual Learning for Image Recognition." In Proceedings of the
%     IEEE conference on computer vision and pattern recognition,
%     pp. 770-778. 2016.
% [2] He, Kaiming, Zhang, Xiangyu, Ren, Shaoqing, and Sun, Jian.
%     "Identity Mappings in Deep Residual Networks." In Computer Vision -
%     ECCV 2016 - 14th European Conference, Amsterdam, The Netherlands,
%     October 11-14, 2016, Proceedings, Part IV, pp. 630-645. 2016.

%   Copyright 2021 The MathWorks, Inc.

%arguments
    %inputSize {mustBeInteger,mustBePositive,iMustBeValidInputSize(inputSize)}
    %numClasses (1,1) {mustBeInteger,mustBePositive}
    options.InitialFilterSize = 7;
    options.InitialNumFilters  = 64;
    options.InitialStride  = [1 2];
    options.InitialPoolingLayer  = 'max';
    options.ResidualBlockType   = 'batchnorm-before-add';
    options.BottleneckType   = 'downsample-first-conv';
    options.StackDepth  = [2];
    options.NumFilters   = [32];
    options.Normalization = 'zerocenter';
%end

%iCheckInputSizeBigEnoughForDepth( inputSize, ...
%     options.InitialStride, ...
%     options.InitialPoolingLayer, ...
%     options.StackDepth );
iMustBeEqualSize(options.NumFilters, options.StackDepth);

lgraph = layerGraph();

initialLayers = iInitialLayers( inputSize, ...
    options.InitialFilterSize, ...
    options.InitialNumFilters, ...
    options.InitialStride, ...
    options.InitialPoolingLayer, ...
    options.Normalization );

lgraph = addLayers(lgraph, initialLayers);

lgraph  = iConvolutionBlockLayers( lgraph, ...
    options.InitialNumFilters, ...
    options.StackDepth, options.NumFilters, ...
    options.BottleneckType, options.ResidualBlockType );

% Connect the input part of the network to the first residual block.
endingLayerOfConv1_x = initialLayers(end).Name;
lgraph = connectLayers(lgraph, endingLayerOfConv1_x, 'stack1_block1_conv1');
if iNumberOfChannelsChangesInInitialBlock(options.InitialNumFilters, ...
        options.NumFilters(1), options.BottleneckType)
    lgraph = connectLayers(lgraph, endingLayerOfConv1_x, 'stack1_block1_skip_conv');
    if strcmp(options.ResidualBlockType, 'batchnorm-before-add')
        lgraph = connectLayers(lgraph, 'stack1_block1_skip_bn', ['stack1_block1_add', '/in2']);
    else
        lgraph = connectLayers(lgraph, 'stack1_block1_skip_conv', ['stack1_block1_add', '/in2']);
    end
else
    lgraph = connectLayers(lgraph, endingLayerOfConv1_x ,['stack1_block1_add' '/in2']);
end

finalLayers = iFinalLayers(numClasses);
lgraph = addLayers(lgraph, finalLayers);

% Connect the last residual block to the end part of the network
suffix = iSuffixOfLastLayerInBlock(options.BottleneckType);
net = connectLayers(lgraph, [ 'stack' num2str(length(options.StackDepth)) '_block' num2str(options.StackDepth(end)) suffix], 'gap');
end

%% Helper functions

function iMustBeValidInputSize(inputSize)
if ~iIsRowVectorOfTwoOrThree(inputSize)
    error(message('nnet_cnn:resnetLayers:InvalidImageSize2D'));
end
end

function tf = iIsRowVectorOfTwoOrThree(x)
tf = isrow(x) && (numel(x)==2 || numel(x)==3);
end

function iMustBeEqualSize(numFilters, stackDepth)
    if ~isequal(size(numFilters),size(stackDepth))
         error(message('nnet_cnn:resnetLayers:NotEqualVectorSizes'));
    end
end

function net = iInitialLayers(inputSize, initialFilterSize, ...
    initialNumOutputChannels, initialStride, initialPoolingLayer, normalization)
poolingLayer = [];
if strcmp(initialPoolingLayer,'max')
    poolingLayer = maxPooling2dLayer([1 3],'Stride',[1 2],'Padding','same','Name','maxpool1');
elseif strcmp(initialPoolingLayer,'average')
    poolingLayer = averagePooling2dLayer([1 3],'Stride',[1 2],'Padding','same','Name','avgpool1');
end

net = [
    imageInputLayer(inputSize,'Normalization', normalization, 'Name','input')
    convolution2dLayer( initialFilterSize, initialNumOutputChannels, ...
        'Stride', initialStride, ...
        'Padding', 'same', ...
        'WeightsInitializer', iDefaultWeightsInitializer(), ...
        'BiasLearnRateFactor', 0, ...
        'Name', 'conv1' )
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    poolingLayer
    ];
end

function net = iFinalLayers(numClasses)
net = [
    globalAveragePooling2dLayer('Name','gap'), ...
    fullyConnectedLayer(numClasses,'Name','fc','WeightsInitializer', iDefaultWeightsInitializer()),...
    softmaxLayer('Name','softmax'), ...
    classificationLayer('Name','output') ];
end

function tf = iNumberOfChannelsChangesInInitialBlock(initialNumFilters, numFilters, bottleneckType)
switch bottleneckType
    case 'none'
        tf = initialNumFilters ~= numFilters;
    case 'downsample-first-conv'
        expansionFactor = 4;
        tf = initialNumFilters ~= numFilters*expansionFactor;
end
end

function layers = iSkipLayers(bottleneckType, residualBlockType, numFilters, stride, tag)
expansionFactor = 4;
if strcmp(bottleneckType, 'downsample-first-conv')
    numFilters = numFilters*expansionFactor;
end
if strcmp(residualBlockType, 'batchnorm-before-add')
    layers = [ convolution2dLayer(1,numFilters,'Stride', stride,'Padding',0, ...
        'WeightsInitializer', iDefaultWeightsInitializer(), ...
        'BiasLearnRateFactor',0, 'Name',[tag 'skip_conv']), ...
        batchNormalizationLayer('Name',[tag 'skip_bn'])];
else
    layers = convolution2dLayer(1,numFilters,'Stride', stride,'Padding',0, ...
        'WeightsInitializer', iDefaultWeightsInitializer(), ...
        'BiasLearnRateFactor',0, 'Name',[tag 'skip_conv']);
end
end

%% Original residual block variant
function layers = iOriginalBasicConvolutionalUnit(numF,stride,tag)
% layers = iOriginalBasicConvolutionalUnit(numF,stride,tag) creates a standard
% convolutional unit, containing two 3-by-3 convolutional layers with numF
% filters and a tag for layer name assignment.
layers = [
    convolution2dLayer(3,numF,'Padding','same','Stride',stride, ...
        'WeightsInitializer',iDefaultWeightsInitializer(), ...
        'BiasLearnRateFactor',0,'Name',[tag,'conv1'])
    batchNormalizationLayer('Name',[tag,'bn1'])
    reluLayer('Name',[tag,'relu1'])
    
    convolution2dLayer(3,numF,'Padding','same', ...
        'WeightsInitializer',iDefaultWeightsInitializer(), ...
        'BiasLearnRateFactor',0,'Name',[tag,'conv2'])
    batchNormalizationLayer('Name',[tag,'bn2'])
    
    additionLayer(2,'Name', [tag, 'add'])
    reluLayer('Name',[tag 'relu2'])];
end

function layers = iOriginalBottleneckConvolutionalUnit(numF,stride,tag)
% layers = iOriginalBottleneckConvolutionalUnit(numF,stride,tag) creates a
% bottleneck convolutional unit, containing two 1-by-1 convolutional layers
% and one 3-by-3 layer. The 3-by-3 layer has numF filters, while the final
% 1-by-1 layer upsamples the output to 4*numF channels. The stride is
% applied in the 3-by-3 convolution so that no input activations are
% completely discarded (the 3-by-3 filters are still overlapping).
expansionFactor = 4;
layers = [
    convolution2dLayer(1,numF,'Padding','same', 'Stride',stride, ...
        'WeightsInitializer',iDefaultWeightsInitializer(), ...
        'BiasLearnRateFactor',0,'Name',[tag,'conv1'])
    batchNormalizationLayer('Name',[tag,'bn1'])
    reluLayer('Name',[tag,'relu1'])
    
    convolution2dLayer(3,numF,'Padding','same', ...
        'WeightsInitializer',iDefaultWeightsInitializer(), ...
        'BiasLearnRateFactor',0,'Name',[tag,'conv2'])
    batchNormalizationLayer('Name',[tag,'bn2'])
    reluLayer('Name',[tag,'relu2'])
    
    convolution2dLayer(1,expansionFactor*numF,'Padding','same', ...
        'WeightsInitializer',iDefaultWeightsInitializer(), ...
        'BiasLearnRateFactor',0,'Name',[tag,'conv3'])
    batchNormalizationLayer('Name',[tag,'bn3'])
    
    additionLayer(2,'Name', [tag, 'add'])
    reluLayer('Name',[tag 'relu3'])];
end

%% "Batch-norm after addition" Residual Block Variant
% This variant of the residual block was proposed in the paper "Identity
% Mappings in Deep Residual Networks".
function layers = iBatchNormAfterAdditionBasicConvolutionalUnit(numF,stride,tag)
layers = [
    convolution2dLayer(3,numF,'Padding','same','Stride',stride, ...
        'WeightsInitializer',iDefaultWeightsInitializer(), ...
        'BiasLearnRateFactor',0,'Name',[tag,'conv1'])
    batchNormalizationLayer('Name',[tag,'bn1'])
    reluLayer('Name',[tag,'relu1'])
    
    convolution2dLayer(3,numF,'Padding','same', ...
        'WeightsInitializer',iDefaultWeightsInitializer(), ...
        'BiasLearnRateFactor',0,'Name',[tag,'conv2'])
    additionLayer(2,'Name', [tag, 'add'])
    
    batchNormalizationLayer('Name',[tag,'bn2'])
    reluLayer('Name',[tag 'relu2'])];
end


function layers = iBatchNormAfterAdditionBottleneckConvolutionalUnit(numF,stride,tag)
expansionFactor = 4;
layers = [
    convolution2dLayer(1,numF,'Padding','same', 'Stride',stride, ...
        'WeightsInitializer',iDefaultWeightsInitializer(), ...
        'BiasLearnRateFactor',0,'Name',[tag,'conv1'])
    batchNormalizationLayer('Name',[tag,'bn1'])
    reluLayer('Name',[tag,'relu1'])
    
    convolution2dLayer(3,numF,'Padding','same', ...
        'WeightsInitializer',iDefaultWeightsInitializer(), ...
        'BiasLearnRateFactor',0,'Name',[tag,'conv2'])
    batchNormalizationLayer('Name',[tag,'bn2'])
    reluLayer('Name',[tag,'relu2'])
    
    convolution2dLayer(1,expansionFactor*numF,'Padding','same', ...
        'WeightsInitializer',iDefaultWeightsInitializer(), ...
        'BiasLearnRateFactor',0,'Name',[tag,'conv3'])
    additionLayer(2,'Name', [tag, 'add'])
    
    batchNormalizationLayer('Name',[tag,'bn3'])       
    reluLayer('Name',[tag 'relu3'])];
end


%% Create main part of Network
function layers = iConvolutionBlockLayers( lgraph, ...
    initialNumFilters, stackDepth, numFilters, bottleneckType, residualBlockType )
layers = lgraph;
for convStackIdx = 1:length(stackDepth) 
    numChannels = numFilters(convStackIdx);
    numBlocks = stackDepth(convStackIdx);
    for blockIdx = 1:numBlocks
        
        S_part = ['stack'  num2str(convStackIdx)];
        U_part = ['_block'  num2str(blockIdx)];
        tag = [S_part U_part '_'];
        
        if blockIdx == 1 && convStackIdx ~= 1
            stride = 2;
        else
            stride = 1;
        end

        residualBlock = iResidualBlockFactory(bottleneckType, residualBlockType, numChannels, stride, tag);       
        layers = addLayers(layers, residualBlock);

        % When to add a skip connection
        if blockIdx == 1 && convStackIdx == 1
            if iNumberOfChannelsChangesInInitialBlock(initialNumFilters, numFilters(1), bottleneckType)
                skip = iSkipLayers(bottleneckType, residualBlockType, numChannels, stride, tag);
                layers = addLayers(layers, skip);
            end
        elseif blockIdx == 1
            skip = iSkipLayers(bottleneckType, residualBlockType, numChannels, stride, tag);
            layers = addLayers(layers, skip);
        end
        
        if blockIdx > 1
            layers = iConnectLayersInResidualBlock(layers, blockIdx, tag, S_part, bottleneckType);
        end
        
    end
       
    if convStackIdx > 1
        S_prev = ['stack'  num2str(convStackIdx - 1)];
        U_prev = ['_block'  num2str(stackDepth(convStackIdx - 1))];
        layers = iConnectBetweenResidualBlocks(layers, S_prev, U_prev, S_part, bottleneckType);
        layers = iConnectSkipLayers(layers, residualBlockType, S_prev, U_prev, S_part, bottleneckType);
    end
 
end
end

function layers = iConnectLayersInResidualBlock(layers, blockIdx, tag, s_part, bottleneckType)
suffix = iSuffixOfLastLayerInBlock(bottleneckType);
U_prev = ['_block'  num2str(blockIdx - 1)];
layers = connectLayers(layers,[s_part U_prev suffix], [tag 'conv1']);
layers = connectLayers(layers,[s_part U_prev suffix], strcat([tag 'add'], '/in2'));
end

function layers = iConnectBetweenResidualBlocks(layers, s_prev, u_prev, s_part, bottleneckType)
% connect from residual block (n - 1) to residual block (n)
suffix = iSuffixOfLastLayerInBlock(bottleneckType);
layers = connectLayers(layers,[s_prev u_prev suffix], [s_part '_block1_' 'conv1']);
end

function layers = iConnectSkipLayers(layers, residualBlockType, s_prev, u_prev, s_part, bottleneckType)
suffix = iSuffixOfLastLayerInBlock(bottleneckType);
layers = connectLayers(layers,[s_prev u_prev suffix] ,[s_part '_block1_' 'skip_conv']);
if strcmp(residualBlockType, 'batchnorm-before-add')
    layers = connectLayers(layers,[s_part '_block1_' 'skip_bn'],strcat([s_part '_block1_' 'add'], '/in2'));
else
    layers = connectLayers(layers,[s_part '_block1_' 'skip_conv'],strcat([s_part '_block1_' 'add'], '/in2'));
end
end

function suffix = iSuffixOfLastLayerInBlock(bottleneckType)
switch bottleneckType
    case 'none'
        suffix = '_relu2';
    case 'downsample-first-conv'
        suffix = '_relu3';
end
end

function residualBlock = iResidualBlockFactory(buildingBlock, residualBlockType, numChannel, stride, tag)
switch residualBlockType
    case 'batchnorm-before-add'
        if strcmp(buildingBlock, 'downsample-first-conv')
            residualBlock = iOriginalBottleneckConvolutionalUnit(numChannel,stride,tag);
        else
            residualBlock = iOriginalBasicConvolutionalUnit(numChannel,stride,tag);
        end
    case 'batchnorm-after-add'
        if strcmp(buildingBlock, 'downsample-first-conv')
            residualBlock = iBatchNormAfterAdditionBottleneckConvolutionalUnit(numChannel,stride,tag);
        else
            residualBlock = iBatchNormAfterAdditionBasicConvolutionalUnit(numChannel,stride,tag);
        end
    otherwise
        error(message('nnet_cnn:resnetLayers:InvalidResidualBlockType',residualBlockType));
end
end

function iCheckInputSizeBigEnoughForDepth(imageSize, initialStride, initialPoolingLayer, stackDepth)
% The network downsamples the image size by 2^depth. Ensure the image
% size doesn't get reduce to less than 1.
depth = length(stackDepth);
if ~strcmp(initialPoolingLayer,'none')
    depth = depth + 1;
end

if any( imageSize(1:2) ./ ((2^depth).*initialStride)  < 1 )
    error(message('nnet_cnn:resnetLayers:ImageTooSmallForDepth',min((2^depth).*initialStride)));
end
end

function initializer = iDefaultWeightsInitializer()
initializer = "he";
end