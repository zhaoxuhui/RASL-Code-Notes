function [fileNames, transformations, numImages] = get_training_images( rootPath, ...
    pointrootPath, trainingDatabaseName, ...
    baseCoords, transformationInit)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   get_training_filenames
%
%   Inputs:
%       rootPath             --
%       trainingDatabaseName --
%       userNameList         --
%       transformationInit   --  initialization transformation type,
%                                depending on how much information on the batch images 
%
%   Outputs:
%       fileNames            --
%       transformations      --
%       labels               --
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Yigang Peng, Arvind Ganesh, November 2009. 
% Questions? abalasu2@illinois.edu
%
% Copyright: Perception and Decision Laboratory, University of Illinois, Urbana-Champaign
%            Microsoft Research Asia, Beijing



transformations = {};
fileNames = {};

imageIndex = 0;

userDirectoryContents = list_image_files(fullfile(rootPath, trainingDatabaseName));

if isempty(userDirectoryContents)
    error(['No image files were found! Check your paths; there should be images in ' fullfile(rootPath, trainingDatabaseName)]);
end

% 如果初始变换是单位阵，那么是不需要对应特征点的
if strcmp(transformationInit,'IDENTITY')
    for fileIndex = 1:length(userDirectoryContents),
        imageName = userDirectoryContents{fileIndex};
        disp(['Using image file ' imageName '...']);

        imageIndex = imageIndex+1;

        imageFileName = fullfile(rootPath, trainingDatabaseName, imageName);
        fileNames{imageIndex} = imageFileName;

        transformations{imageIndex} = [1, 0, 0; 0 1 0; 0 0 1] ;
    end
elseif strcmp(transformationInit,'SIMILARITY') % 相似变换
    for fileIndex = 1:length(userDirectoryContents),
        imageName = userDirectoryContents{fileIndex};
        disp(['Using image file ' imageName '...']);

        imageIndex = imageIndex+1;

        imageFileName = fullfile(rootPath, trainingDatabaseName, imageName);
        fileNames{imageIndex} = imageFileName;

        pointFileName = fullfile(pointrootPath, trainingDatabaseName, imageName);
        % Load the initial control point data. 加载初始控制点数据
        CornerFileName = [pointFileName(1:end-4) '-points.mat'];
        % 如果没有角点数据，就直接退出了
        if exist(CornerFileName, 'file'),
            load(CornerFileName); 
        else
            error(['No corner data found for image ' imageFileName '!']);
        end
        
        % points就是读取的角点文件的内容
        % points的每一列为一个坐标点
        % 将基准点与对应的点对应，计算相似矩阵S
        % S的最后一列为平移分量，第一行为dx，第二行为dy
        S = TwoPointSimilarity( baseCoords, points(:,1:2) );
        S
        % 将求解出来的相似变换矩阵作为后续迭代的初值放到列表里
        transformations{imageIndex} = S ;
    end
elseif strcmp(transformationInit,'AFFINE') % 仿射变换，和上面相似变换的步骤是一样的，唯一区别是对应点数量
    for fileIndex = 1:length(userDirectoryContents),
        imageName = userDirectoryContents{fileIndex};
        disp(['Using image file ' imageName '...']);

        imageIndex = imageIndex+1;

        imageFileName = fullfile(rootPath, trainingDatabaseName, imageName);
        fileNames{imageIndex} = imageFileName;

        pointFileName = fullfile(pointrootPath, trainingDatabaseName, imageName);
        % Load the initial control point data.
        CornerFileName = [pointFileName(1:end-4) '-points.mat'];
        if exist(CornerFileName, 'file'),
            load(CornerFileName); 
        else
            error(['No corner data found for image ' imageFileName '!']);
        end
        
        % 相比于相似变换，这里加载了3个坐标点。points的每一列为一个坐标点
        % 将基准点与对应的点对应，计算仿射变换矩阵S
        % 由于不同的影像对应了不同坐标点，所以每张影像求解出来的仿射矩阵也不相同
        % s11 s12 s13
        % s21 s22 s23
        % 对应变换后的坐标x'、y'可以计算：
        % x'=s11*x+s12*y+s13
        % y'=s21*x+s22*y+s23
        % 注意这里求解出来的仿射变换矩阵顺序是从baseCoords->points
        S = ThreePointAffine( baseCoords, points(:,1:3) );
        S
        transformations{imageIndex} = S ;   % 将仿射变换矩阵添加进列表
    end
else
    error(['unable to initialize the transformations!']);
end

numImages = length(userDirectoryContents) ;
return;
