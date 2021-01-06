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
    
    points_training{i} = detectSURFFeatures(images{i});
    
%     imshow(images{i}); hold on;
%     plot(points{i}); hold off;
    
    A = [double(points_training{i}.Scale), double(points_training{i}.SignOfLaplacian), ...
        double(points_training{i}.Orientation), double(points_training{i}.Location), ...
        double(points_training{i}.Metric)];
    
    SURF_features = [SURF_features; A];

end

HiddenLayerSize=4;
autoenc = trainAutoencoder(SURF_features', HiddenLayerSize, ...
    'ScaleData', true, ...
    'UseGPU', true, ...
    'SparsityRegularization',1.6);


%%
clc;

delete('features\*.txt');

for i = 1:numel(imdsTest.Files)
    I = rgb2gray(readimage(imdsTraining, i));
    
    images_test{i} = I;  %store images in the cell array
    points_test{i} = detectSURFFeatures(images_test{i});
    
    A =[double(points_test{i}.Scale), ...
        double(points_test{i}.SignOfLaplacian), ...
        double(points_test{i}.Orientation), ...
        double(points_test{i}.Location), ...
        double(points_test{i}.Metric)];
    
    Y_test = predict(autoenc, A')';
    
    initial = 0+1;
    final = length(Y_test);
    
    Scale = Y_test(initial:final , 1);
    SignOfLaplacian = Y_test(initial:final, 2);
    Orientation = Y_test(initial:final, 3);
    Location = Y_test(initial:final, 4:5);
    Metric = Y_test(initial:final, 6);

%     points_autoenc{i} = SURFPoints(Location, ...
%         'Scale', single(Scale), ...
%         'SignOfLaplacian', int8(SignOfLaplacian), ...
%         'Orientation', single(Orientation), ...
%         'Metric', single(Metric));
    points_autoenc{i} = SURFPoints(Location, ...
        'Scale', single(Scale), ...
        'Orientation', single(Orientation));
    
    imshow(images_test{i}); hold on;
    plot(points_autoenc{i}); hold off;
    
    
    A = [points_autoenc{i}.Location, ...
        points_autoenc{i}.Scale, points_autoenc{i}.Orientation];
    for j=1:128
        A=[A, zeros(length(A),1)];
    end
    [~,name,ext] = fileparts(string(imdsTest.Files{i}));
    dlmwrite('features/' + string(name)+string(ext) +'.txt', A);
    
end
