function DogSkinApp
% Dog Skin Disease Prediction GUI using trained DenseNet model

    clc;

    % Load trained model (make sure this .mat is in the same folder)
    data = load('DogSkinDenseNet201.mat','trainedNet','inputSize','classes');
    net = data.trainedNet;
    inputSize = data.inputSize;
    classes = data.classes; %#ok<NASGU> % not used but kept for reference

    % Create main window
    fig = uifigure('Name','Dog Skin Disease Classifier', ...
                   'Position',[150 100 950 550]);

    % Image display panel
    ax = uiaxes(fig,'Position',[25 120 380 380]);
    title(ax,'Dog Skin Image');
    axis(ax,'off');
    %ax = uiaxes(fig,'Position',[25 120 380 380]);


    % Button to load image
    btnLoad = uibutton(fig,'push', ...
        'Text','Load Image from Disk', ...
        'FontSize',13, ...
        'Position',[430 470 200 40], ...
        'ButtonPushedFcn',@(btn,event)loadAndPredict());

    % Result label
    lblResult = uilabel(fig, ...
        'Position',[430 430 480 30], ...
        'Text','Prediction: ---', ...
        'FontSize',15, ...
        'FontWeight','bold');

    % Advice box
    txtAdvice = uitextarea(fig, ...
        'Position',[430 100 480 300], ...
        'Value',{'Owner advice will appear here...'}, ...
        'Editable','off', ...
        'FontSize',12);

    % -------- Callback Function --------
    function loadAndPredict()

        [file, path] = uigetfile({'*.jpg;*.jpeg;*.png','Image Files'});
        if isequal(file,0)
            return;
        end

        filename = fullfile(path,file);
        img = imread(filename);

        % Show original image
        imshow(img,'Parent',ax);
        axis(ax,'image');
        title(ax,'Selected Image');

        % Ensure RGB
        if size(img,3) == 1
            img = cat(3,img,img,img);
        end

        % Resize to network input
        imgResized = imresize(img, inputSize(1:2));

        % Predict
        [label, scores] = classify(net, imgResized);
        confidence = max(scores)*100;
        disease = string(label);

        % Display result (simple)
        lblResult.Text = sprintf('Prediction: %s (%.2f%%)', disease, confidence);

        % Display advice
        advice = getOwnerAdvice(disease);
        txtAdvice.Value = advice;

        % Draw lesion box
        drawBoundingBox(ax, img, disease, confidence);
    end
end


%% ------- Bounding Box Function -------
function drawBoundingBox(ax, img, label, confidence)
    % Draws a red box around the largest suspicious region

    % Ensure RGB
    if size(img,3) == 1
        img = cat(3,img,img,img);
    end

    % Convert to HSV
    hsvImg = rgb2hsv(img);
    S = hsvImg(:,:,2);
    V = hsvImg(:,:,3);

    % Detect suspicious area (darker + more saturated)
    lesionMask = (S > 0.25) & (V < 0.75);

    % Clean mask
    lesionMask = imclose(lesionMask, strel('disk',5));
    lesionMask = bwareaopen(lesionMask, 400);
    lesionMask = imfill(lesionMask, 'holes');

    % Find largest region
    regionStats = regionprops(lesionMask, 'BoundingBox', 'Area');

    if ~isempty(regionStats)
        [~, idx] = max([regionStats.Area]);
        box = regionStats(idx).BoundingBox;
    else
        % Fallback: center of image
        h = size(img,1);
        w = size(img,2);
        box = [w/4 h/4 w/2 h/2];
    end

    % Display image + bounding box
    imshow(img,'Parent',ax);
    axis(ax,'image');
    hold(ax,'on');
    rectangle(ax,'Position',box,'EdgeColor','r','LineWidth',2);
    title(ax,sprintf('%s (%.2f%%)', string(label), confidence));
    hold(ax,'off');
end
