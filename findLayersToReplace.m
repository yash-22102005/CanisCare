function [learnableLayer, classLayer] = findLayersToReplace(lgraph)
% findLayersToReplace Helper function to find the last learnable layer
% and the classification layer in a LayerGraph (works for many pretrained nets)

if ~isa(lgraph, 'nnet.cnn.LayerGraph')
    error('Input must be a LayerGraph');
end

layers      = lgraph.Layers;
connections = lgraph.Connections;

% Find classification output layer
idx = arrayfun(@(l) isa(l,'nnet.cnn.layer.ClassificationOutputLayer'), layers);
classLayer = layers(idx);

if numel(classLayer) ~= 1
    error('Network must have a single classification output layer.');
end

% Find source layer that feeds into the classification layer
srcLayerName  = connections.Source(strcmp(connections.Destination, classLayer.Name));
learnableLayer = layers(strcmp({layers.Name}, srcLayerName));

% Check learnable type
if ~isa(learnableLayer, 'nnet.cnn.layer.FullyConnectedLayer') && ...
   ~isa(learnableLayer, 'nnet.cnn.layer.Convolution2DLayer')
    error('The layer feeding the classification layer is not a learnable layer.');
end
end
