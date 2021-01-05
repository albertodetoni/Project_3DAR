clear all; close all; clc;

imdsTraining = imageDatastore(...
    'Camera_Calibration\Training', 'IncludeSubfolders', 1);

imdsTest = imageDatastore(...
    'Camera_Calibration\Testing', 'IncludeSubfolders', 1);

% dim=250; % dimension all the images will have
SURF_features=[];
for i = 1:numel(imdsTraining.Files) % for each image
    I = rgb2gray(readimage(imdsTraining, i)); % store and convert to gray the image
    
%     if size(I,1)>size(I,2) % padding the images to be all squares
%         I = imresize(I, [dim NaN]);
%         longdim = size(I,2);
%         I = padarray(I, [0 floor((dim-longdim)/2)], 0, 'pre');
%         I = padarray(I, [0 ceil((dim-longdim)/2)], 0, 'post');
%     else
%         I = imresize(I, [NaN dim]);
%         longdim = size(I,1);
%         I = padarray(I, [floor((dim-longdim)/2) 0], 0, 'pre');
%         I = padarray(I, [ceil((dim-longdim)/2) 0], 0, 'post');
%     end
    
    
    images{i} = I;  %store images in the cell array
    
    points{i} = detectSURFFeatures(images{i});
    
%     imshow(images{i}); hold on;
%     plot(points{i}); hold off;
    
    A = double([points{i}.Scale, points{i}.SignOfLaplacian, ...
        points{i}.Orientation, points{i}.Location, points{i}.Metric]);
    
    SURF_features = [SURF_features; A];

end

HiddenLayerSize=15;
autoenc = trainAutoencoder(SURF_features', HiddenLayerSize, ...
    'ScaleData', true, ...
    'UseGPU', true);

%%
clc;

SURF_features_test=[]; surf_feat_number=[];
for i = 1:numel(imdsTest.Files)
    I = rgb2gray(readimage(imdsTraining, i));
    
    images_test{i} = I;  %store images in the cell array
    points_test{i} = detectSURFFeatures(images_test{i});
    A = double([points_test{i}.Scale, points_test{i}.SignOfLaplacian, ...
        points_test{i}.Orientation, points_test{i}.Location, ...
        points_test{i}.Metric]);
    
    SURF_features_test = [SURF_features_test; A];
    surf_feat_number = [surf_feat_number; length(A)];
    
end

Y_test = predict(autoenc, SURF_features_test')';


surf_feat_number = [0; surf_feat_number];
for i=1:length(points_test)-1
    points_test{i}.Scale = Y_test(surf_feat_number(i)+1 : surf_feat_number(i+1), 1);
    points_test{i}.SignOfLaplacian = Y_test(surf_feat_number(i)+1 : surf_feat_number(i+1), 2);
    points_test{i}.Orientation = Y_test(surf_feat_number(i)+1 : surf_feat_number(i+1), 3);
    points_test{i}.Location = Y_test(surf_feat_number(i)+1 : surf_feat_number(i+1), 4:5);
    points_test{i}.Metric = Y_test(surf_feat_number(i)+1 : surf_feat_number(i+1), 6);
    
    imshow(images_test{i}); hold on;
    plot(points_test{i}); hold off;
end

