% Yigang Peng, Arvind Ganesh, November 2009. 
% Questions? abalasu2@illinois.edu
%
% Copyright: Perception and Decision Laboratory, University of Illinois, Urbana-Champaign
%            Microsoft Research Asia, Beijing



function xi = projective_matrix_to_parameters( transformType, T )
xi = [];
if strcmp(transformType,'TRANSLATION'), % 平移变换
    xi = T(1:2,3); % 对于平移变换类型，只提取变换矩阵最后一列的前两行，对应x、y分量
elseif strcmp(transformType,'EUCLIDEAN'),   % 欧氏刚体变换
    xi = nan(3,1);  % 建立了一个3×1的向量
    % 求取旋转角度和方向
    theta = acos(T(1,1));
    if T(2,1) < 0,
        theta = -theta;
    end
    xi(1) = theta;  % 旋转角度(弧度)
    xi(2) = T(1,3); % x平移分量
    xi(3) = T(2,3); % y平移分量
elseif strcmp(transformType,'SIMILARITY'),  % 相似变换
    xi = nan(4,1);
    sI = T(1:2,1:2)' * T(1:2,1:2);
    xi(1) = sqrt(sI(1));    % 缩放参数
    theta = acos(T(1,1)/xi(1));
    if T(2,1) < 0,
        theta = -theta;
    end
    xi(2) = theta;  % 旋转角度(弧度)
    xi(3) = T(1,3); % x平移分量
    xi(4) = T(2,3); % y平移分量
elseif strcmp(transformType,'AFFINE'),  % 仿射变换
    xi = nan(6,1);
    xi(1:3) = T(1,:)';  % 变换矩阵的第一行
    xi(4:6) = T(2,:)';  % 变换矩阵的第二行
    % 因此，xi中参数的顺序是r11、r12、dx、r21、r22、dy
elseif strcmp(transformType,'HOMOGRAPHY'),  % 单应变换
    xi = nan(8,1);
    xi(1:3) = T(1,:)';  % 变换矩阵的第一行
    xi(4:6) = T(2,:)';  % 变换矩阵的第二行
    xi(7:8) = T(3,1:2)';    % 变换矩阵第三行的前两列
else
    error('Unrecognized transformation');
end