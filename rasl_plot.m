function rasl_plot(destDir, numImage, canonicalImageSize, layout)

%% load in data
% initial input images ��ʼ������Ӱ������D
load(fullfile(destDir, 'original.mat'), 'D') ;

% alignment results ����֮��Ľ��
load(fullfile(destDir, 'final.mat'), 'Do','A','E') ;

%% output data files
length = size(D);
length = length(2);
for i=1:length
    tmp_ori_img = reshape(D(:,i),[canonicalImageSize(1),canonicalImageSize(2)]);
    tmp_align_img = reshape(Do(:,i),[canonicalImageSize(1),canonicalImageSize(2)]);
    tmp_low_img = reshape(A(:,i),[canonicalImageSize(1),canonicalImageSize(2)]);
    tmp_sparse_img = reshape(E(:,i),[canonicalImageSize(1),canonicalImageSize(2)]);
    imwrite(mat2gray(tmp_ori_img),fullfile(destDir,strcat(num2str(i,'%03d'),'_original.png')));
    imwrite(mat2gray(tmp_align_img),fullfile(destDir,strcat(num2str(i,'%03d'),'_align.png')));
    imwrite(mat2gray(tmp_low_img),fullfile(destDir,strcat(num2str(i,'%03d'),'_low.png')));
    imwrite(mat2gray(tmp_sparse_img),fullfile(destDir,strcat(num2str(i,'%03d'),'_sparse.png')));
end

%% display

% layout �������
if nargin < 4
    xI = ceil(sqrt(numImage)) ;
    yI = ceil(numImage/xI) ;

    gap = 2;
    gap2 = 1; % gap2 = gap/2;
else
    xI = layout.xI ;
    yI = layout.yI ;

    gap = layout.gap ;
    gap2 = layout.gap2 ; % gap2 = gap/2;
end
container = ones(canonicalImageSize(1)+gap, canonicalImageSize(2)+gap); 
% white edges
bigpic = cell(xI,yI); % (xI*canonicalImageSize(1),yI*canonicalImageSize(2));

% D
for i = 1:xI
    for j = 1:yI
        if yI*(i-1)+j > numImage
            bigpic{i,j} = ones(canonicalImageSize(1)+gap, canonicalImageSize(2)+gap);
        else
            container ((gap2+1):(end-gap2), (gap2+1):(end-gap2)) = reshape(D(:,yI*(i-1)+j), canonicalImageSize);
            bigpic{i,j} = container;
        end
    end
end
figure
imshow(cell2mat(bigpic),[],'DisplayRange',[0 max(max(D))],'Border','tight')
title('Input images') ;

% Do
for i = 1:xI
    for j = 1:yI
        if yI*(i-1)+j > numImage
            bigpic{i,j} = ones(canonicalImageSize(1)+gap, canonicalImageSize(2)+gap);
        else
            container ((gap2+1):(end-gap2), (gap2+1):(end-gap2)) = reshape(Do(:,yI*(i-1)+j), canonicalImageSize);
            bigpic{i,j} = container;
        end
    end
end
figure
imshow(cell2mat(bigpic),[],'DisplayRange',[0 max(max(Do))],'Border','tight')
title('Aligned images') ;


% A
for i = 1:xI
    for j = 1:yI
        if yI*(i-1)+j > numImage
            bigpic{i,j} = ones(canonicalImageSize(1)+gap, canonicalImageSize(2)+gap);
        else
            container ((gap2+1):(end-gap2), (gap2+1):(end-gap2)) = reshape(A(:,yI*(i-1)+j), canonicalImageSize);
            bigpic{i,j} = container;
        end
    end
end
figure
imshow(cell2mat(bigpic),[],'DisplayRange',[0 max(max(A))],'Border','tight')
title('Aligned images adjusted for sparse errors') ;

% E
for i = 1:xI
    for j = 1:yI
        if yI*(i-1)+j > numImage
            bigpic{i,j} = ones(canonicalImageSize(1)+gap, canonicalImageSize(2)+gap);
        else
            container ((gap2+1):(end-gap2), (gap2+1):(end-gap2)) = reshape(E(:,yI*(i-1)+j), canonicalImageSize);
            bigpic{i,j} = container;
        end
    end
end
figure
imshow(abs(cell2mat(bigpic)),[],'DisplayRange',[0 max(max(abs(E)))],'Border','tight')
title('Sparse corruptions in the aligned images') ;

% average face of D, Do and A
% bigpic = cell(1,3); 
% container ((gap2+1):(end-gap2), (gap2+1):(end-gap2)) = reshape(sum(D,2), canonicalImageSize);
% bigpic{1,1} = container;
% container ((gap2+1):(end-gap2), (gap2+1):(end-gap2)) = reshape(sum(Do,2), canonicalImageSize);
% bigpic{1,2} = container;
% container ((gap2+1):(end-gap2), (gap2+1):(end-gap2)) = reshape(sum(A,2), canonicalImageSize);
% bigpic{1,3} = container;
% 
figure
subplot(1,3,1)
imshow(reshape(sum(D,2), canonicalImageSize),[])
title('average of unaligned D')
subplot(1,3,2)
imshow(reshape(sum(Do,2), canonicalImageSize),[])
title('average of aligned D')
subplot(1,3,3)
imshow(reshape(sum(A,2), canonicalImageSize),[])
title('average of A')