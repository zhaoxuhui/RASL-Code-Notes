% Yigang Peng, Arvind Ganesh, November 2009. 
% Questions? abalasu2@illinois.edu
%
% Copyright: Perception and Decision Laboratory, University of Illinois, Urbana-Champaign
%            Microsoft Research Asia, Beijing

% Computes projective matrix based on input parameters and transformation
% type.

function T = parameters_to_projective_matrix( transformType, xi )
T = eye(3); % 先建立一个3×3的单位阵，之后向其中填充内容
if strcmp(transformType,'TRANSLATION'), % 平移变换，2个参数
    % 对于平移变换，将变换矩阵T的最后一列的前两行填充dx、dy
    T(1,3) = xi(1);
    T(2,3) = xi(2);
elseif strcmp(transformType,'EUCLIDEAN'), % 欧氏刚体变换，3个参数
    % 先构建2×2的旋转矩阵
    R = [ cos(xi(1)), -sin(xi(1)); ...
        sin(xi(1)), cos(xi(1)) ];
    T(1:2,1:2) = R;
    % 再将平移填充
    T(1,3) = xi(2);
    T(2,3) = xi(3);
elseif strcmp(transformType,'SIMILARITY'), % 相似变换，4个参数
    % 旋转矩阵
    R = [ cos(xi(2)), -sin(xi(2)); ...
        sin(xi(2)), cos(xi(2)) ];
    % 再乘以缩放系数
    T(1:2,1:2) = xi(1)*R;
    % 平移部分
    T(1,3) = xi(3);
    T(2,3) = xi(4);
elseif strcmp(transformType,'AFFINE'), % 仿射变换，6个参数
    % 由于保存的时候直接就是按行存的，所以这里就十分简单了
    T(1:2,:) = [ xi(1:3)'; xi(4:6)' ];
elseif strcmp(transformType,'HOMOGRAPHY'), % 单应变换，8个参数
    % 和仿射变换类似
    T = [ xi(1:3)'; xi(4:6)'; [xi(7:8)' 1] ];
else
    % 其它变换认为是不识别的变换
    error('Unrecognized transformation');
end