% Yigang Peng, Arvind Ganesh, November 2009. 
% Questions? abalasu2@illinois.edu
%
% Copyright: Perception and Decision Laboratory, University of Illinois, Urbana-Champaign
%            Microsoft Research Asia, Beijing

% Computes projective matrix based on input parameters and transformation
% type.

function T = parameters_to_projective_matrix( transformType, xi )
T = eye(3); % �Ƚ���һ��3��3�ĵ�λ��֮���������������
if strcmp(transformType,'TRANSLATION'), % ƽ�Ʊ任��2������
    % ����ƽ�Ʊ任�����任����T�����һ�е�ǰ�������dx��dy
    T(1,3) = xi(1);
    T(2,3) = xi(2);
elseif strcmp(transformType,'EUCLIDEAN'), % ŷ�ϸ���任��3������
    % �ȹ���2��2����ת����
    R = [ cos(xi(1)), -sin(xi(1)); ...
        sin(xi(1)), cos(xi(1)) ];
    T(1:2,1:2) = R;
    % �ٽ�ƽ�����
    T(1,3) = xi(2);
    T(2,3) = xi(3);
elseif strcmp(transformType,'SIMILARITY'), % ���Ʊ任��4������
    % ��ת����
    R = [ cos(xi(2)), -sin(xi(2)); ...
        sin(xi(2)), cos(xi(2)) ];
    % �ٳ�������ϵ��
    T(1:2,1:2) = xi(1)*R;
    % ƽ�Ʋ���
    T(1,3) = xi(3);
    T(2,3) = xi(4);
elseif strcmp(transformType,'AFFINE'), % ����任��6������
    % ���ڱ����ʱ��ֱ�Ӿ��ǰ��д�ģ����������ʮ�ּ���
    T(1:2,:) = [ xi(1:3)'; xi(4:6)' ];
elseif strcmp(transformType,'HOMOGRAPHY'), % ��Ӧ�任��8������
    % �ͷ���任����
    T = [ xi(1:3)'; xi(4:6)'; [xi(7:8)' 1] ];
else
    % �����任��Ϊ�ǲ�ʶ��ı任
    error('Unrecognized transformation');
end