%% TrainDogSkinModel.m
% End-to-end training script for Dog Skin Disease Classification
% Uses DenseNet-201 + transfer learning + data augmentation.

clc;
clear;
close all;

%% 1. Dataset Path

% Change this ONLY if your folder is different
folderPath = 'C:\Users\yash2\OneDrive\Desktop\DOG SKIN DISEASES';

%% 2. Create ImageDatastore

imds = imageDatastore(folderPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

disp('--- Dataset Summary ---');
tbl = countEachLabel(imds);
disp(tbl);

numClasses = numel(categories(imds.Labels));
%% Handle Class Imbalance using Class Weights

totalImages = sum(tbl.Count);
alpha = 0.5;   % Weight softness factor (0.4–0.7 works well)
classWeights = (totalImages ./ (numClasses * tbl.Count)).^alpha;


disp('Class Weights:');
disp(classWeights);

% Store class names
classNames = tbl.Label;

% Create weighted classification layer
newClassLayer = classificationLayer( ...
    'Name', 'new_classoutput', ...
    'Classes', classNames, ...
    'ClassWeights', classWeights);

fprintf('Number of disease classes detected: %d\n', numClasses);

%% 3. Split Data into Train / Validation / Test
% 70% Train, 15% Validation, 15% Test (per class)

[imdsTrain, imdsTemp] = splitEachLabel(imds, 0.7, 'randomized');  % 70%
[imdsVal, imdsTest]   = splitEachLabel(imdsTemp, 0.5, 'randomized'); % 15/15

fprintf('\n--- Data Split Summary ---\n');
fprintf('Training images:   %d\n', numel(imdsTrain.Files));
fprintf('Validation images: %d\n', numel(imdsVal.Files));
fprintf('Testing images:    %d\n', numel(imdsTest.Files));

%% 4. Load Pretrained Network (DenseNet-201)

net = densenet201;   % Requires Deep Learning Toolbox Model for DenseNet-201
inputSize = net.Layers(1).InputSize;
fprintf('\nBase network: DenseNet-201. Input size: [%d %d %d]\n', inputSize);

%% 5. Data Augmentation and Resizing

imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandRotation', [-10 10], ...
    'RandScale', [0.9 1.1] ...
);




% Augmented training datastore
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter);

% Validation & Test: only resizing
augimdsVal  = augmentedImageDatastore(inputSize(1:2), imdsVal);
augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest);

%% 6. Modify Network for Transfer Learning (DenseNet-201 FIX)

lgraph = layerGraph(net);
% Freeze first N layers safely (DO NOT rebuild graph)
layerNames = {lgraph.Layers.Name};
freezeTill = layerNames(1:150);   % First 150 layers frozen

for i = 1:numel(freezeTill)
    layer = lgraph.Layers(strcmp(layerNames, freezeTill{i}));
    if isprop(layer,'WeightLearnRateFactor')
        layer.WeightLearnRateFactor = 0;
        layer.BiasLearnRateFactor   = 0;
        lgraph = replaceLayer(lgraph, layer.Name, layer);
    end
end



% Replace final fully connected layer (DenseNet-201 specific)
newFCLayer = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);

lgraph = replaceLayer(lgraph, 'fc1000', newFCLayer);

% Replace classification layer
% newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph, net.Layers(end).Name, newClassLayer);


%% 7. Training Options

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 16, ...
    'MaxEpochs', 35, ...
    'InitialLearnRate', 3e-5, ...
    'Momentum', 0.9, ...
    'L2Regularization', 0.0005, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsVal, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 12, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

fprintf('\n--- Starting Training ---\n');
trainedNet = trainNetwork(augimdsTrain, lgraph, options);
fprintf('\n--- Training Complete ---\n');

%% 8. Evaluation on Test Set

fprintf('\n--- Evaluating on Test Set ---\n');

YPred = classify(trainedNet, augimdsTest);
YTest = imdsTest.Labels;

accuracy = mean(YPred == YTest) * 100;
fprintf('Final Test Accuracy: %.2f%%\n', accuracy);

figure;
confusionchart(YTest, YPred);
title(sprintf('Confusion Matrix (Accuracy = %.2f%%)', accuracy));

%% 9. Save Trained Model

classes = categories(imds.Labels);
save('DogSkinDenseNet201.mat', 'trainedNet', 'inputSize', 'classes');

fprintf('\nModel saved to DogSkinDenseNet201.mat\n');

