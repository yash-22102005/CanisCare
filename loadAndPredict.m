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

    % Predict class
    [label, scores] = classify(net, imgResized);
    confidence = max(scores)*100;
    disease = string(label);

    % Display result
    lblResult.Text = sprintf('Prediction: %s (%.2f%%)', disease, confidence);

    % Display owner advice
    advice = getOwnerAdvice(disease);
    txtAdvice.Value = advice;

    % Draw lesion bounding box
    drawBoundingBox(ax, img, disease, confidence);

    % -------- FEATURE EXTRACTION DISPLAY --------
    try
        layerName = 'conv1';
        features = activations(net, imgResized, layerName);
        features = mat2gray(features);
        montage(features,'Parent',axFeature);
        title(axFeature,'CNN Feature Maps (Conv1)');
    catch
        cla(axFeature);
        text(axFeature,0.1,0.5,'Feature map unavailable','Color','r');
    end

end
