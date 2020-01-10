% Yigang Peng, Arvind Ganesh, November 2009. 
% Questions? abalasu2@illinois.edu
%
% Copyright: Perception and Decision Laboratory, University of Illinois, Urbana-Champaign
%            Microsoft Research Asia, Beijing
%
% Reference: RASL: Robust Alignment by Sparse and Low-rank Decomposition for Linearly Correlated Images  
%            Yigang Peng, Arvind Ganesh, John Wright, Wenli Xu, and Yi Ma. Proc. of CVPR, 2010.
%

% Figure 7 in the paper
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

currentPath = cd;

% input path
imagePath = fullfile(currentPath,'data') ;
pointPath = fullfile(currentPath,'data') ; % path to files containing initial feature coordinates
userName = 'rs6' ;

% output path
destRoot = fullfile(currentPath,'results') ;
destDir = fullfile(destRoot,userName) ;
if ~exist(destDir,'dir')
    mkdir(destRoot,userName) ;
end

%% define parameters

% dispaly flag 是否展示每次外层迭代后的结果
raslpara.DISPLAY = 1 ;

% save flag
raslpara.saveStart = 1 ;
raslpara.saveEnd = 1 ;
raslpara.saveIntermedia = 0 ;

% for windows images
raslpara.canonicalImageSize = [ 400 400  ]; % 影像的大小，下面这三个点都会在这个范围内
raslpara.canonicalCoords = [ 326  349  41; ...  % 每一列为一个坐标点，共3个，分别是(26,24)、(176,24)、(100,184)
                             140  367   277 ];
% 每一个影像都有三个对应点，配合这里的三个点，就可以依次求出从这三个点变换到每一张影像的变换关系，以这个关系作为初始变换关系

                            
% parametric tranformation model
raslpara.transformType = 'HOMOGRAPHY'; 
% one of 'TRANSLATION', 'EUCLIDEAN', 'SIMILARITY', 'AFFINE','HOMOGRAPHY'

raslpara.numScales = 1 ; % if numScales > 1, we use multiscales

% main loop 主循环相关参数
raslpara.stoppingDelta = .01; % stopping condition of main loop 外层循环的迭代终止条件
raslpara.maxIter = 25; % maximum iteration number of main loops 外层循环的最大迭代次数

% inner loop 内层循环相关参数
raslpara.inner_tol = 1e-6 ;
raslpara.inner_maxIter = 1000 ;
raslpara.continuationFlag = 1 ;
raslpara.mu = 1e-3 ;
raslpara.lambdac = 1 ; % lambda = lambdac/sqrt(m) 一般情况下都设为1


%% Get training images

% get initial transformation
transformationInit = 'AFFINE';  % 仿射变换需要3对点计算

[fileNames, transformations, numImages] = get_training_images( imagePath, pointPath, userName, raslpara.canonicalCoords, transformationInit) ;


%% RASL main loop: do robust batch image alignment

[D, Do, A, E, xi, numIterOuter, numIterInner ] = rasl_main(fileNames, transformations, numImages, raslpara, destDir);

%% plot the results

layout.xI = 4 ;
layout.yI = 4 ;
layout.gap = 4 ;
layout.gap2 = 2 ;
rasl_plot(destDir, numImages, raslpara.canonicalImageSize, layout)
