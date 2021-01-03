clear all; close all; clc;

imdsTraining = imageDatastore(...
    'Camera_Calibration\Training', 'IncludeSubfolders', 1);

imdsTest = imageDatastore(...
    'Camera_Calibration\Testing', 'IncludeSubfolders', 1);
h=figure;
dim=250; % dimension all the images will have
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
    
    imshow(images{i}); hold on;
    plot(points{i}); hold off;
    
    A = [points{i}.Scale, points{i}.SignOfLaplacian, points{i}.Orientation, points{i}.Location, points{i}.Metric];
    A=double(A);
    HiddenLayerSize=15;
    autoenc = trainAutoencoder(A, HiddenLayerSize);
%     autoenc = trainAutoencoder(points{i}.Location, HiddenLayerSize);

end


%%
clc;
for i = 1:numel(imdsTest.Files)
    I = rgb2gray(readimage(imdsTraining, i));
    I=double(I);
    
    Y_test{i} = predict(autoenc, I);
end