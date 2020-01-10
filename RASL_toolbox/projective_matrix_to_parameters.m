% Yigang Peng, Arvind Ganesh, November 2009. 
% Questions? abalasu2@illinois.edu
%
% Copyright: Perception and Decision Laboratory, University of Illinois, Urbana-Champaign
%            Microsoft Research Asia, Beijing



function xi = projective_matrix_to_parameters( transformType, T )
xi = [];
if strcmp(transformType,'TRANSLATION'), % ƽ�Ʊ任
    xi = T(1:2,3); % ����ƽ�Ʊ任���ͣ�ֻ��ȡ�任�������һ�е�ǰ���У���Ӧx��y����
elseif strcmp(transformType,'EUCLIDEAN'),   % ŷ�ϸ���任
    xi = nan(3,1);  % ������һ��3��1������
    % ��ȡ��ת�ǶȺͷ���
    theta = acos(T(1,1));
    if T(2,1) < 0,
        theta = -theta;
    end
    xi(1) = theta;  % ��ת�Ƕ�(����)
    xi(2) = T(1,3); % xƽ�Ʒ���
    xi(3) = T(2,3); % yƽ�Ʒ���
elseif strcmp(transformType,'SIMILARITY'),  % ���Ʊ任
    xi = nan(4,1);
    sI = T(1:2,1:2)' * T(1:2,1:2);
    xi(1) = sqrt(sI(1));    % ���Ų���
    theta = acos(T(1,1)/xi(1));
    if T(2,1) < 0,
        theta = -theta;
    end
    xi(2) = theta;  % ��ת�Ƕ�(����)
    xi(3) = T(1,3); % xƽ�Ʒ���
    xi(4) = T(2,3); % yƽ�Ʒ���
elseif strcmp(transformType,'AFFINE'),  % ����任
    xi = nan(6,1);
    xi(1:3) = T(1,:)';  % �任����ĵ�һ��
    xi(4:6) = T(2,:)';  % �任����ĵڶ���
    % ��ˣ�xi�в�����˳����r11��r12��dx��r21��r22��dy
elseif strcmp(transformType,'HOMOGRAPHY'),  % ��Ӧ�任
    xi = nan(8,1);
    xi(1:3) = T(1,:)';  % �任����ĵ�һ��
    xi(4:6) = T(2,:)';  % �任����ĵڶ���
    xi(7:8) = T(3,1:2)';    % �任��������е�ǰ����
else
    error('Unrecognized transformation');
end