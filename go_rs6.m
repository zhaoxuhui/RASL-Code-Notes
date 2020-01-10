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

% dispaly flag �Ƿ�չʾÿ����������Ľ��
raslpara.DISPLAY = 1 ;

% save flag
raslpara.saveStart = 1 ;
raslpara.saveEnd = 1 ;
raslpara.saveIntermedia = 0 ;

% for windows images
raslpara.canonicalImageSize = [ 400 400  ]; % Ӱ��Ĵ�С�������������㶼���������Χ��
raslpara.canonicalCoords = [ 326  349  41; ...  % ÿһ��Ϊһ������㣬��3�����ֱ���(26,24)��(176,24)��(100,184)
                             140  367   277 ];
% ÿһ��Ӱ����������Ӧ�㣬�������������㣬�Ϳ��������������������任��ÿһ��Ӱ��ı任��ϵ���������ϵ��Ϊ��ʼ�任��ϵ

                            
% parametric tranformation model
raslpara.transformType = 'HOMOGRAPHY'; 
% one of 'TRANSLATION', 'EUCLIDEAN', 'SIMILARITY', 'AFFINE','HOMOGRAPHY'

raslpara.numScales = 1 ; % if numScales > 1, we use multiscales

% main loop ��ѭ����ز���
raslpara.stoppingDelta = .01; % stopping condition of main loop ���ѭ���ĵ�����ֹ����
raslpara.maxIter = 25; % maximum iteration number of main loops ���ѭ��������������

% inner loop �ڲ�ѭ����ز���
raslpara.inner_tol = 1e-6 ;
raslpara.inner_maxIter = 1000 ;
raslpara.continuationFlag = 1 ;
raslpara.mu = 1e-3 ;
raslpara.lambdac = 1 ; % lambda = lambdac/sqrt(m) һ������¶���Ϊ1


%% Get training images

% get initial transformation
transformationInit = 'AFFINE';  % ����任��Ҫ3�Ե����

[fileNames, transformations, numImages] = get_training_images( imagePath, pointPath, userName, raslpara.canonicalCoords, transformationInit) ;


%% RASL main loop: do robust batch image alignment

[D, Do, A, E, xi, numIterOuter, numIterInner ] = rasl_main(fileNames, transformations, numImages, raslpara, destDir);

%% plot the results

layout.xI = 4 ;
layout.yI = 4 ;
layout.gap = 4 ;
layout.gap2 = 2 ;
rasl_plot(destDir, numImages, raslpara.canonicalImageSize, layout)
