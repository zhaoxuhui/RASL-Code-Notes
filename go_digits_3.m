% Yigang Peng, Arvind Ganesh, November 2009. 
% Questions? abalasu2@illinois.edu
%
% Copyright: Perception and Decision Laboratory, University of Illinois, Urbana-Champaign
%            Microsoft Research Asia, Beijing
%
% Reference: RASL: Robust Alignment by Sparse and Low-rank Decomposition for Linearly Correlated Images  
%            Yigang Peng, Arvind Ganesh, John Wright, Wenli Xu, and Yi Ma. Proc. of CVPR, 2010.
%

% Figure 6 in the paper
% robust batch image alignment example

% clear
clc ;
clear all ;
close all ;

% addpath
addpath RASL_toolbox ;
addpath data ;
addpath results ;

%% define images' path

currentPath = cd ;

% input path fullfile��������ƴ��·��
imagePath = fullfile(currentPath, 'data') ;
pointPath = fullfile(currentPath, 'data') ;
userName = 'Digits_3' ;

% output path
destRoot = fullfile(currentPath,'results') ;
destDir = fullfile(destRoot,userName) ;
if ~exist(destDir,'dir')
    mkdir(destRoot,userName) ;
end

%% define parameters

% dispaly flag
raslpara.DISPLAY = 1 ;

% save flag
raslpara.saveStart = 1 ;
raslpara.saveEnd = 1 ;
raslpara.saveIntermedia = 0 ;

% for windows images
raslpara.canonicalImageSize = [ 29  29];    % Ϊʲô��29 29��
raslpara.canonicalCoords = [ 5  24 ; ...    % ����������ʼ����㣬�ֱ���(5,15)��(24,15)
                             15 15  ];
                            
% parametric tranformation model
raslpara.transformType = 'EUCLIDEAN';   % �任����
% one of 'TRANSLATION', 'EUCLIDEAN', 'SIMILARITY', 'AFFINE','HOMOGRAPHY'

raslpara.numScales = 1 ; % if numScales > 1, we use multiscales

% main loop
raslpara.stoppingDelta = .01; % stopping condition of main loop
raslpara.maxIter = 25; % maximum iteration number of main loops

% inner loop
raslpara.inner_tol = 1e-6 ;
raslpara.inner_maxIter = 1000 ;
raslpara.continuationFlag = 1 ;
raslpara.mu = 1e-3 ;
raslpara.lambdac = 1 ; % lambda = lambdac/sqrt(m)


%% Get training images

% get initial transformation ��ʼ�任��Ϣ
transformationInit = 'SIMILARITY';  % ���Ʊ任��Ҫ2�Ե����
[fileNames, transformations, numImages] = get_training_images( imagePath, pointPath, userName, raslpara.canonicalCoords, transformationInit) ;


%% RASL main loop: do robust batch image alignment

[D, Do, A, E, xi, numIterOuter, numIterInner ] = rasl_main(fileNames, transformations, numImages, raslpara, destDir);

%% plot the results

layout.xI = 10 ;
layout.yI = 10 ;
layout.gap = 0 ;
layout.gap2 = 0 ;
rasl_plot(destDir, numImages, raslpara.canonicalImageSize, layout)
