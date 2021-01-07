%% TRAINING

clear all; close all; clc;

imdsTraining = imageDatastore(...
    'images\Training', 'IncludeSubfolders', 1);

imdsTest = imageDatastore(...
    'images\Testing', 'IncludeSubfolders', 1);

SURF_features=[];
for i = 1:numel(imdsTraining.Files) % for each image
    I = rgb2gray(readimage(imdsTraining, i)); % store and convert to gray the image
    
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

save('Workspace_autoenc_trained.mat');

%% TESTING
clc; clear all;

load('Workspace_autoenc_trained.mat')

delete('features/tiso/*.txt'); delete('features/fountain/*.txt');

for i = 1:numel(imdsTest.Files)
    I = rgb2gray(readimage(imdsTest, i));
    
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

    points_autoenc{i} = SURFPoints(Location, ...
        'Scale', single(Scale), ...
        'Orientation', single(Orientation));
    
    imshow(images_test{i}); hold on;
    plot(points_autoenc{i}); hold off;
    
    A = [points_autoenc{i}.Location, ...
        points_autoenc{i}.Scale, points_autoenc{i}.Orientation];
    for j=1:128
        A = [A, zeros(length(A),1)];
    end
    
    [pth,name,ext] = fileparts(string(imdsTest.Files{i}));

    if contains(pth,'fountain','IgnoreCase',true)
        filePath = 'features/fountain/' + string(name)+string(ext) +'.txt';
    elseif contains(pth,'tiso','IgnoreCase',true)
        filePath = 'features/tiso/' + string(name)+string(ext) +'.txt';
    end
    
    fileID = fopen(filePath, 'w');
    fprintf(fileID, string(length(A))+' 128\n');
    fclose(fileID);
    
    writematrix(A, filePath, 'WriteMode', 'append', 'Delimiter', 'space');
    
end
