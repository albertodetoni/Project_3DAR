clear all;
close all;

flag_col=0;

warning off;

% Load all the images to be used in the dataset
dir_files = 'images';
imds = imageDatastore(dir_files);

image_order=[ 1:11];

imds.Files = imds.Files(image_order);

%rmdir('dataSift','s');
%mkdir('dataSift')


% Display the images.
figure
montage(imds.Files, 'Size', [2, 4]);

% Convert the images to grayscale.
images = cell(1, numel(imds.Files));
for i = 1:numel(imds.Files)
    I = readimage(imds, i);
    images{i} = rgb2gray(I);  %store images in the cell array
end

Npic=numel(imds.Files);  %number of images
desc_arc = cell(1,Npic);
keypoints_arc = cell(1,Npic);


%Compute descriptors and range of features
for i = 1:numel(images)
    % Undistort the current image.
    I = images{i};
    
    % Detect features. 
    prevPoints   = detectSURFFeatures(I, 'NumOctaves', 4,'MetricThreshold',50) ;
    keypoints_arc{i}=prevPoints;
    % Extract features. Using 'Upright' features improves matching
    prevFeatures = extractFeatures(I, prevPoints, 'Upright', true);
    desc_arc{i}=prevFeatures;
end;

title('Input Image Sequence');





 str_sfm='sfmData';
    if exist(str_sfm, 'dir')
        [sr,sc]=system(sprintf('rm -fr %s',str_sfm));
    end;
    mkdir(str_sfm);
    mkdir([ str_sfm '/img' ]);
    mkdir([ str_sfm '/sift' ]);

for i=1:numel(images)
    ppp=keypoints_arc{i};
    fff=desc_arc{i};
    fff=[ zeros(size(fff)) zeros(size(fff)) ];
    
    
    str0=char(imds.Files(i));
    istr0=max(strfind(str0,'/'));
    str=[ str_sfm '/img/' str0(istr0+1:end)];
    imwrite(images{i},str);
    str_txt=strrep(strrep(str ,'jpg','jpg.txt'),'img','sift');
    
    fp=fopen(str_txt,'w');
    ccc=fprintf(fp,'%d 128\n',size(fff,1));
    for c=1:size(fff,1)
        ccc=ppp(c).Location;
        ccc=fprintf(fp,'%f %f %f %f',ccc(1),ccc(2),ppp(c).Scale,ppp(c).Orientation);
        ccc=fprintf(fp,' %d',fff(c,:));
        ccc=fprintf(fp,'\n');
    end;
    fclose(fp);
end;

fp=fopen(sprintf('%s/mmm_file.txt',str_sfm),'w');
for c1=1:numel(images)
    prevFeatures = desc_arc{c1};
    prevPoints   = keypoints_arc{c1};
    
    str1=char(imds.Files(c1));
    istr1=max(strfind(str1,'/'));
    str1=str1(istr1+1:end);
    
    for c2=setdiff(1:numel(images),c1)
        currFeatures = desc_arc{c2};
        currPoints   = keypoints_arc{c2};
        
        str2=char(imds.Files(c2));
        istr2=max(strfind(str2,'/'));
        str2=str2(istr2+1:end);
        
        indexPairs0 = matchFeatures(prevFeatures, currFeatures, ...
                'MaxRatio', .7, 'Unique',  true);
        
        if (size(indexPairs0,1)>0)
            ccc=fprintf(fp,'%s %s\n',str1,str2);
            for m=1:size(indexPairs0,1)
                ccc=fprintf(fp,'%d %d\n',indexPairs0(m,1)-1,indexPairs0(m,2)-1);
            end;
            ccc=fprintf(fp,'\n');
        end;
    end;
end;
 fclose(fp);