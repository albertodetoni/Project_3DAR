clear all; close all; clc;

imds = imageDatastore(...
    'Camera_Calibration\Testing\tisoDataset');

dim=250; % dimension all the images will have
for i = 1:numel(imds.Files) % for each image
    I = rgb2gray(readimage(imds, i)); % store and convert to gray the image
    
    if size(I,1)>size(I,2) % padding the images to be all squares
        I = imresize(I, [dim NaN]);
        longdim = size(I,2);
        I = padarray(I, [0 floor((dim-longdim)/2)], 0, 'pre');
        I = padarray(I, [0 ceil((dim-longdim)/2)], 0, 'post');
    else
        I = imresize(I, [NaN dim]);
        longdim = size(I,1);
        I = padarray(I, [floor((dim-longdim)/2) 0], 0, 'pre');
        I = padarray(I, [ceil((dim-longdim)/2) 0], 0, 'post');
    end
    
    
    images{i} = I;  %store images in the cell array
    
    points{i} = detectSURFFeatures(images{i});
    
    imshow(images{i}); hold on;
    plot(points{i}.selectStrongest(10)); hold off;  
    
end

HiddenLayerSize=50;
autoenc = trainAutoencoder(images, HiddenLayerSize);
