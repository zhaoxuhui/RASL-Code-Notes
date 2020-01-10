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

% �����ʼ�任�ǵ�λ����ô�ǲ���Ҫ��Ӧ�������
if strcmp(transformationInit,'IDENTITY')
    for fileIndex = 1:length(userDirectoryContents),
        imageName = userDirectoryContents{fileIndex};
        disp(['Using image file ' imageName '...']);

        imageIndex = imageIndex+1;

        imageFileName = fullfile(rootPath, trainingDatabaseName, imageName);
        fileNames{imageIndex} = imageFileName;

        transformations{imageIndex} = [1, 0, 0; 0 1 0; 0 0 1] ;
    end
elseif strcmp(transformationInit,'SIMILARITY') % ���Ʊ任
    for fileIndex = 1:length(userDirectoryContents),
        imageName = userDirectoryContents{fileIndex};
        disp(['Using image file ' imageName '...']);

        imageIndex = imageIndex+1;

        imageFileName = fullfile(rootPath, trainingDatabaseName, imageName);
        fileNames{imageIndex} = imageFileName;

        pointFileName = fullfile(pointrootPath, trainingDatabaseName, imageName);
        % Load the initial control point data. ���س�ʼ���Ƶ�����
        CornerFileName = [pointFileName(1:end-4) '-points.mat'];
        % ���û�нǵ����ݣ���ֱ���˳���
        if exist(CornerFileName, 'file'),
            load(CornerFileName); 
        else
            error(['No corner data found for image ' imageFileName '!']);
        end
        
        % points���Ƕ�ȡ�Ľǵ��ļ�������
        % points��ÿһ��Ϊһ�������
        % ����׼�����Ӧ�ĵ��Ӧ���������ƾ���S
        % S�����һ��Ϊƽ�Ʒ�������һ��Ϊdx���ڶ���Ϊdy
        S = TwoPointSimilarity( baseCoords, points(:,1:2) );
        S
        % �������������Ʊ任������Ϊ���������ĳ�ֵ�ŵ��б���
        transformations{imageIndex} = S ;
    end
elseif strcmp(transformationInit,'AFFINE') % ����任�����������Ʊ任�Ĳ�����һ���ģ�Ψһ�����Ƕ�Ӧ������
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
        
        % ��������Ʊ任�����������3������㡣points��ÿһ��Ϊһ�������
        % ����׼�����Ӧ�ĵ��Ӧ���������任����S
        % ���ڲ�ͬ��Ӱ���Ӧ�˲�ͬ����㣬����ÿ��Ӱ���������ķ������Ҳ����ͬ
        % s11 s12 s13
        % s21 s22 s23
        % ��Ӧ�任�������x'��y'���Լ��㣺
        % x'=s11*x+s12*y+s13
        % y'=s21*x+s22*y+s23
        % ע�������������ķ���任����˳���Ǵ�baseCoords->points
        S = ThreePointAffine( baseCoords, points(:,1:3) );
        S
        transformations{imageIndex} = S ;   % ������任������ӽ��б�
    end
else
    error(['unable to initialize the transformations!']);
end

numImages = length(userDirectoryContents) ;
return;
