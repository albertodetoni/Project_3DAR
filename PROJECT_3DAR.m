%% TRAINING

clear all; close all; clc;

imdsTraining = imageDatastore(...
    'images\Training', 'IncludeSubfolders', 1);

imdsTest = imageDatastore(...
    'images\Testing', 'IncludeSubfolders', 1);

SURF_features=[];
for i = 1:numel(imdsTraining.Files) % for each image
    I = rgb2gray(readimage(imdsTraining, i)); % convert to gray the image
    images{i} = I;  %store images in the cell array
    
    [points_training{i}, valpoints] = extractFeatures(images{i}, detectSURFFeatures(images{i}, 'NumOctaves', 4, 'MetricThreshold', 50), 'Upright', true);
    
    imshow(images{i}); hold on;
    plot(valpoints); hold off;
    
    A = double(points_training{i});
    
    SURF_features = [SURF_features; A];
end

close all;
HiddenLayerSize=4;
autoenc = trainAutoencoder(SURF_features', HiddenLayerSize, ...
    'UseGPU', false);
%    'ScaleData', true);

save('Workspace_autoenc_trained.mat');

%% TESTING
%% FEATURE EXTRACTION

clc; clear all;
load('Workspace_autoenc_trained.mat')

imdsFountain = imageDatastore(...
    'images\Testing\fountain-P11', 'IncludeSubfolders', true);

imdsTiso = imageDatastore(...
    'images\Testing\tisoDataset', 'IncludeSubfolders', true);


features_fountain = FEATURES(imdsFountain, autoenc);
%features_tiso = FEATURES(imdsTiso, autoenc);
close all;

%% MATCHINGS
clc;

MATCHINGS(imdsFountain, features_fountain);
%MATCHINGS(imdsTiso, features_fountain);

%% FUNCTIONS

function features = FEATURES (imds, autoenc)
    
    delete('features/tiso/*.txt'); delete('features/fountain/*.txt');
    delete('features/portello/*.txt'); delete('features/castle/*.txt');

    for i = 1:numel(imds.Files)
        I = rgb2gray(readimage(imds, i));
        images{i} = I;  %store images in the cell array
        
        [descriptors{i}, valpoints] = extractFeatures(images{i}, detectSURFFeatures(images{i}, 'NumOctaves', 4, 'MetricThreshold', 50), 'Upright', true);

        A=double(descriptors{i});

        Y_test = predict(autoenc, A')'; % WITH AUTOENCODER
        %Y_test = A; % WITHOUT AUTOENCODER
        
        %plot of the images%%%
        imshow(images{i}); hold on;
        plot(valpoints); hold off;
        %%%%%%%%%%%%%%%%%%%%%%
%         
        A =[double(valpoints.Location), ...
            double(valpoints.Scale), ...
            double(valpoints.Orientation)];
        for j=1:128
            A = [A, zeros(length(A),1)];
        end

        [pth,name,ext] = fileparts(string(imds.Files{i}));
        if contains(pth,'fountain','IgnoreCase',true)
            filePath = 'features/fountain/' + string(name)+string(ext) +'.txt';
            
        elseif contains(pth,'tiso','IgnoreCase',true)
            filePath = 'features/tiso/' + string(name)+string(ext) +'.txt';
            
        elseif contains(pth,'portello','IgnoreCase',true)
            filePath = 'features/portello/' + string(name)+string(ext) +'.txt';
            
        elseif contains(pth,'castle','IgnoreCase',true)
            filePath = 'features/castle/' + string(name)+string(ext) +'.txt';
        end
        features{i} = single(Y_test);
        features = features(~cellfun('isempty',features));

        fileID = fopen(filePath, 'w');
        fprintf(fileID, string(length(A))+' 128\n');
        fclose(fileID);

        writematrix(A, filePath, 'WriteMode', 'append', 'Delimiter', 'space');

    end
end


function MATCHINGS (imds, features)

    delete('matchings/tiso/*.txt'); delete('matchings/fountain/*.txt');
    delete('matchings/portello/*.txt'); delete('matchings/castle/*.txt');         
    
    for i=1:length(features)-1
        for j=i+1:length(features)
            matchings = matchFeatures(features{i},features{j}, 'MaxRatio', .7, 'Unique',  true);

            [pth,name_from,ext_from] = fileparts(string(imds.Files{i}));
            [~,name_to,ext_to] = fileparts(string(imds.Files{j}));

            if contains(pth, 'tiso', 'IgnoreCase', true)
                filePath = 'matchings/tiso/matchings.txt';
            elseif contains(pth, 'fountain', 'IgnoreCase', true)
                filePath = 'matchings/fountain/matchings.txt';
            elseif contains(pth, 'castle', 'IgnoreCase', true)
                filePath = 'matchings/castle/matchings.txt';
            elseif contains(pth, 'portello', 'IgnoreCase', true)
                filePath = 'matchings/portello/matchings.txt';
            end

            title = string(name_from)+string(ext_from) +' '+ string(name_to)+string(ext_to);

            fileID = fopen(filePath, 'a');
            fprintf(fileID, title);
            fclose(fileID);

            writematrix(matchings, filePath, 'WriteMode', 'append', 'Delimiter', 'space');

            fileID = fopen(filePath, 'a');
            fprintf(fileID, '\n');
            fclose(fileID);
        end
    end

end