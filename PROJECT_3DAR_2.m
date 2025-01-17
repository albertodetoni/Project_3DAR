%% TRAINING
clear all; close all; clc;

imdsTraining = imageDatastore(...
    'images\Training', 'IncludeSubfolders', 1);

SURF_features=[];
for i = 1:numel(imdsTraining.Files) % for each image

    images{i} = rgb2gray(readimage(imdsTraining, i));  
    
    [points_training{i}, valpoints] = extractFeatures(images{i}, ...
        detectSURFFeatures(images{i}, 'NumOctaves', 4, ...
        'MetricThreshold', 50), 'Upright', true);
    
    imshow(images{i}); hold on;
    plot(valpoints); hold off;
    
    A = double(points_training{i});
    
    SURF_features = [SURF_features; A];
end

close all; clc;

layers = [
    featureInputLayer(64,'Name','IN')  
    fullyConnectedLayer(6,'Name','mid2')
    fullyConnectedLayer(64, 'Name', 'mid4')
    regressionLayer( 'Name', 'OUT')];

lgraph = layerGraph(layers);

figure; plot(lgraph);

options = trainingOptions('adam', ...
    'MaxEpochs',2, ...
    'Shuffle','never', ...
    'Verbose',false, ...
    'Plots','training-progress');

compressed=SURF_features;
autoenc = trainNetwork(SURF_features,compressed,lgraph,options);

save('Workspace_autoenc_trained.mat');

%% TESTING

clc; clear all; load('Workspace_autoenc_trained.mat');

imdsPortello = imageDatastore(...
    'images\Training\portelloDataset', 'IncludeSubfolders', true);

imdsCastle = imageDatastore(...
    'images\Training\castle', 'IncludeSubfolders', true);

imdsFountain = imageDatastore(...
    'images\Testing\fountain-P11', 'IncludeSubfolders', true);

imdsTiso = imageDatastore(...
    'images\Testing\tisoDataset', 'IncludeSubfolders', true);


% MATCHINGS(imdsFountain,  FEATURES(imdsFountain, autoenc));
MATCHINGS(imdsTiso, FEATURES(imdsTiso, autoenc));


%% FUNCTIONS

function features = FEATURES (imds, autoenc)
    
    delete('features/tiso/*.txt'); delete('features/fountain/*.txt');
    delete('features/portello/*.txt'); delete('features/castle/*.txt'); 

    for i = 1:numel(imds.Files)
       
        images{i} = rgb2gray(readimage(imds, i));  
        
        [descriptors{i}, valpoints] = extractFeatures(images{i}, ...
            detectSURFFeatures(images{i}, 'NumOctaves', 4, ...
            'MetricThreshold', 50), 'Upright', true);

        A=double(descriptors{i});

        Y_test = predict(autoenc, A); % WITH AUTOENCODER
%         Y_test = A; % WITHOUT AUTOENCODER
        
        %plot of the images%%%
        imshow(images{i}); hold on;
        plot(valpoints); hold off;
        %%%%%%%%%%%%%%%%%%%%%%
         
        A =[double(valpoints.Location), ...
            double(valpoints.Scale), ...
            double(valpoints.Orientation), ...
            zeros(length(valpoints.Location),128)];

        [path,name,ext] = fileparts(string(imds.Files{i}));
        if contains(path,'fountain','IgnoreCase',true)
            filePath = 'features/fountain/' + string(name)+string(ext) ...
                +'.txt';
            
        elseif contains(path,'tiso','IgnoreCase',true)
            filePath = 'features/tiso/' + string(name)+string(ext) ...
                +'.txt';
            
        elseif contains(path,'portello','IgnoreCase',true)
            filePath = 'features/portello/' + string(name)+string(ext) ...
                +'.txt';
            
        elseif contains(path,'castle','IgnoreCase',true)
            filePath = 'features/castle/' + string(name)+string(ext) ...
                +'.txt';
        end
        
        features{i} = single(Y_test);

        fileID = fopen(filePath, 'w');
        fprintf(fileID, string(length(A))+' 128\n');
        fclose(fileID);

        writematrix(A, filePath, 'WriteMode', 'append', ...
            'Delimiter', 'space');
        
    end
    
    close all;
end



function MATCHINGS (imds, features)

    delete('matchings/tiso/matchings.txt'); 
    delete('matchings/fountain/matchings.txt');
    delete('matchings/portello/matchings.txt');
    delete('matchings/castle/matchings.txt'); clc;        
    
    for i=1:length(features)-1
        for j=i+1:length(features)
            
            matchings = matchFeatures(features{i}, features{j}, ...
                'MaxRatio', .9, 'Unique',  true, 'Method', 'Approximate')-1;

            [path,name_from,ext_from] = fileparts(string(imds.Files{i}));
            [~,name_to,ext_to] = fileparts(string(imds.Files{j}));

            if contains(path, 'tiso', 'IgnoreCase', true)
                filePath = 'matchings/tiso/matchings.txt';
            elseif contains(path, 'fountain', 'IgnoreCase', true)
                filePath = 'matchings/fountain/matchings.txt';
            elseif contains(path, 'castle', 'IgnoreCase', true)
                filePath = 'matchings/castle/matchings.txt';
            elseif contains(path, 'portello', 'IgnoreCase', true)
                filePath = 'matchings/portello/matchings.txt';
            end

            title = string(name_from)+string(ext_from) +' '+ ...
                string(name_to)+string(ext_to) + '\n';

            fileID = fopen(filePath, 'a');
            fprintf(fileID, title);

            writematrix(matchings, filePath, 'WriteMode', 'append', ...
                'Delimiter', 'space');

            fprintf(fileID, '\n');
            fclose(fileID);
            
        end
    end
    
end