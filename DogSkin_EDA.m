%% Dog Skin Disease EDA (Enhanced with Heatmap & Boxplot)
clc; clear; close all;

folderPath = 'C:\Users\yash2\OneDrive\Desktop\DOG SKIN DISEASES';
imds = imageDatastore(folderPath,'IncludeSubfolders',true,'LabelSource','foldernames');

tbl = countEachLabel(imds);
disp('Dataset Summary');
disp(tbl);

%% 1. CLASS DISTRIBUTION
figure;
bar(tbl.Label, tbl.Count);
title('Class Distribution of Dog Skin Diseases');
xlabel('Disease Category');
ylabel('Number of Images');
grid on;

%% 2. SAMPLE IMAGES FROM EACH CLASS
figure;
labels = categories(imds.Labels);

for i = 1:length(labels)
    idx = find(imds.Labels == labels(i), 1);
    img = readimage(imds, idx);
    subplot(2,2,i);
    imshow(img);
    title(labels{i});
end
sgtitle('Sample Images per Class');

%% 3. IMAGE SIZE DISTRIBUTION (BOXPLOT)
numImages = numel(imds.Files);
sizes = zeros(numImages,2);

for i = 1:numImages
    info = imfinfo(imds.Files{i});
    sizes(i,:) = [info.Width info.Height];
end

figure;
boxplot(sizes, 'Labels',{'Width','Height'});
title('Image Size Distribution');
ylabel('Pixels');

%% 4. RGB HISTOGRAM
img = readimage(imds,5);

figure;
subplot(3,1,1); imhist(img(:,:,1)); title('Red Channel');
subplot(3,1,2); imhist(img(:,:,2)); title('Green Channel');
subplot(3,1,3); imhist(img(:,:,3)); title('Blue Channel');

%% 5. RANDOM SAMPLE GRID
figure;
for i = 1:6
    img = readimage(imds,randi(numel(imds.Files)));
    subplot(2,3,i);
    imshow(img);
    title('Random Sample');
end
sgtitle('Random Dataset Images');


%% 6. FEATURE EXTRACTION FOR CORRELATION HEATMAP
disp('Extracting statistical features for correlation heatmap...');

features = zeros(numImages,5);

for i = 1:numImages
    img = imread(imds.Files{i});

    if size(img,3)==3
        gray = rgb2gray(img);
        R = img(:,:,1);
        G = img(:,:,2);
        B = img(:,:,3);
    else
        gray = img;
        R = img; G = img; B = img;
    end

    features(i,1) = mean(gray(:));         % Brightness
    features(i,2) = std(double(gray(:)));  % Contrast
    features(i,3) = mean(R(:));             % Avg Red
    features(i,4) = mean(G(:));             % Avg Green
    features(i,5) = mean(B(:));             % Avg Blue
end

%% Feature Table
featureNames = {'Brightness','Contrast','Red','Green','Blue'};
featureTable = array2table(features,'VariableNames',featureNames);

%% CORRELATION HEATMAP
corrMatrix = corrcoef(features);

figure;
heatmap(featureNames, featureNames, corrMatrix, ...
    'Colormap', parula, ...
    'ColorLimits', [-1 1]);

title('Correlation Heatmap of Image Features');

%% FEATURE BOXPLOT
figure;
boxplot(features, 'Labels', featureNames);
title('Feature Value Distribution');
ylabel('Value');

disp('EDA COMPLETE');
