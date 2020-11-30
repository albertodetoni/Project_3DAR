clear all; close all; clc;

files = dir('Camera_Calibration\Testing\fountain-P11\gt_dense_cameras\*.jpg.camera');

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');

imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');