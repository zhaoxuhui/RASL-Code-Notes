function [D, Do, A, E, xi, numIterOuter, numIterInner ] = rasl_main(fileNames, transformations, numImages, raslpara, destDir)

% ---------------------------------------------------
% Batch image alignment: RASL main loop
%
% input: fileNames               --- the names of images 影像的名称
%        transformations         --- the initial transformation matrix 初始变换矩阵
%        numImages               --- the number of images 影像的个数
%        raslpara                --- parameters for RASL RASL参数
%        destDir                 --- output 输出路径
%
% output: D                      --- input images in canonical frame 原始坐标系中的影像
%         Do                     --- output aligned images 输出的对齐影像
%         A                      --- low-rank component 低秩部分
%         E                      --- sparse error component 稀疏部分
%         xi                     --- transformation parameters 变换参数
%         numIterOuter           --- number of outer loop iterations 外层迭代次数
%         numIterInner           --- total number of inner loop iterations 内层迭代次数
% ---------------------------------------------------

%% read and store full images

APGorALM_flag = 1;
% APGorALM_flag: 1: APG algorithm
%                0: inexact ALM algorithm  

fixGammaType = 1 ;

if ~fixGammaType
    if exist(fullfile(rootPath, userName, 'gamma_is_ntsc'), 'file')
        gammaType = 'ntsc' ;
    elseif exist(fullfile(rootPath, userName, 'gamma_is_srgb'), 'file')
        gammaType = 'srgb' ;
    elseif exist(fullfile(rootPath, userName, 'gamma_is_linear'), 'file')
        gammaType = 'linear' ;
    else
        error('Gamma type not specified for training database!  Please create a file of the form gamma_is_*') ;
    end
else
    gammaType = 'linear' ;
end

sigma0 = 2/5 ;
sigmaS = 1 ;

deGammaTraining = true ;

% cell(m,n)函数用于建立一个m×n的空的数组(矩阵)
I0 = cell(raslpara.numScales,numImages) ; % images 输入的没有经过任何处理的影像
I0x = cell(raslpara.numScales,numImages) ; % image derivatives 输入的原始影像对应的x、y方向的梯度影像
I0y = cell(raslpara.numScales,numImages) ;

for fileIndex = 1 : numImages
    % 以double形式读取当前文件名所对应的影像
    currentImage = double(imread(fileNames{fileIndex}));
    
    % 这里的size函数第一个参数是对象，第二个参数是对象的维度
    % 例如A是一个4×5的矩阵，size(A,1)就表示A的第一维的长度，也就是4
    % Use only the green channel in case of color images    
    if size(currentImage,3) > 1,   currentImage = currentImage(:,:,2);            end
    if deGammaTraining,      currentImage = gamma_decompress(currentImage, gammaType); end
    
    % 建立高斯金字塔
    currentImagePyramid = gauss_pyramid( currentImage, raslpara.numScales,...
        sqrt(det(transformations{fileIndex}(1:2,1:2)))*sigma0, sigmaS );
    
    % 这是一个for循环，用来迭代不同尺度，从numScales开始到1结束，步长为-1
    for scaleIndex = raslpara.numScales:-1:1
        I0{scaleIndex,fileIndex} = currentImagePyramid{scaleIndex};
        
        % image derivatives 对影像求导
        % fspecial函数用于创建一个预定义的滤波算子
        % imfilter函数用于对影像按指定算子滤波
        I0_smooth = I0{scaleIndex,fileIndex};
        I0x{scaleIndex,fileIndex} = imfilter( I0_smooth, (-fspecial('sobel')') / 8 );
        I0y{scaleIndex,fileIndex} = imfilter( I0_smooth,  -fspecial('sobel')   / 8 );
    end
end



%% get the initial input images in canonical frame

imgSize = raslpara.canonicalImageSize ; 

xi_initial = cell(1,numImages) ; % initial transformation parameters 用于存放初始变换参数
for i = 1 : numImages
    % 对输入的变换矩阵进行判断，如果变换矩阵的行数小于3，则在其后面追加一行0 0 1
    % 在get_training_images函数里除了IDENTITY返回的是一个3×3的单位阵，SIMILARITY和AFFINE返回的都是2×3的矩阵
    if size(transformations{i},1) < 3
        transformations{i} = [transformations{i} ; 0 0 1] ;
    end
    % 经过上一步以后，得到的就是一个标准的3×3的二维影像的变换矩阵了，再将其转换为变换参数
    % 对于RASL而言，只支持特定的几种变换类型如TRANSLATION、AFFINE、SIMILARITY、HOMOGRAPHY、EUCLIDEAN，对于其它类型就认为是未识别的了
    % 事实上上面这几种变换对于整体的二维变换已经足够了，但对于可能存在局部畸变的遥感影像而言，同样也是无能为力
    % 对于不同类型的变换，xi的参数个数也不同：
    % TRANSLATION 2个参数，dx、dy
    % EUCLIDEAN   3个参数，旋转角度(弧度)、x平移分量、y平移分量
    % SIMILARITY  4个参数，缩放参数、旋转角度(弧度)、x平移分量、y平移分量
    % AFFINE      6个参数，r11、r12、dx、r21、r22、dy
    % HOMOGRAPHY  8个参数，h11、h12、h13、h21、h22、h23、h31、h32
    xi_initial{i} = projective_matrix_to_parameters(raslpara.transformType,transformations{i});
end

%D = [] ;
D= zeros(imgSize(1)*imgSize(2), numImages);
for fileIndex = 1 : numImages
    % transformed image 
    Tfm = fliptform(maketform('projective',transformations{fileIndex}'));
    
    % 对影像按照初始变换进行变换，重采内插方式是bicubic-双三次插值
    I   = vec(imtransform(I0{1,fileIndex}, Tfm,'bicubic','XData',[1 imgSize(2)],'YData',[1 imgSize(1)],'Size',imgSize));
    y   = I; 
    y = y / norm(y) ; % normalize 将影像的灰度归一化

    %D = [D y] ;
    D(:,fileIndex) = y ;    % 将按照初始变换后的、灰度归一化的影像追加到D中
end

% 如果设置了保存初始状态，就将D(按照初始变换后的、灰度归一化的影像)和xi_initial(初始变换参数)保存到original.mat
% 需要注意的是，这里保存的是按照初始变换矩阵变换过的影像，而不是直接读取的影像
if raslpara.saveStart
    save(fullfile(destDir, 'original.mat'),'D','xi_initial');
end

% 整个上面的步骤其实就是RASL用于解决较大幅度的misalignment的办法
% 先用一个初始变换将差异较大的影像给变换重采到差不多对齐的状态，基于此状态再进行迭代
% 之所以说差不多对齐的状态是因为基于选择的对应点对构建的初始状态不一定严格、精确


%% start the main loop 开始RASL的主循环

frOrig = cell(1,numImages) ;
T_in = cell(1,numImages) ;

T_ds = [ 0.5,   0, -0.5; ...
         0,   0.5, -0.5   ];
T_ds_hom = [ T_ds; [ 0 0 1 ]];

numIterOuter = 0 ;  % 外层循环总的迭代次数
numIterInner = 0 ;  % 内层循环总的迭代次数

tic % time counting start

for scaleIndex = raslpara.numScales:-1:1 % multiscale 用于解决多尺度问题，这里的尺度设为1，所以只执行1次
    
    iterNum = 0 ;  % iteration number of outer loop in each scale 每个尺度的外层循环次数
    converged = 0 ; % flag变量，迭代是否收敛
    prevObj = inf ; % previous objective function value 上一次迭代的目标函数值
    
    % 由于尺度是1，所以这里imgSize就等于raslpara.canonicalImageSize
    imgSize = raslpara.canonicalImageSize / 2^(scaleIndex-1) ;    
    xi = cell(1,numImages) ;
    
    % 由于考虑到尺度问题，对初始变换逐个进行处理，将每个变换矩阵转换一下变成T_in
    for fileIndex = 1 : numImages
        % scaleIndex为1，raslpara.numScales也为1，所以执行if语句
        if scaleIndex == raslpara.numScales
            % 由于并没有涉及到尺度问题，scaleIndex为1，所以这里的T_in和输入的初始transformations相同
            T_in{fileIndex} = T_ds_hom^(scaleIndex-1)*transformations{fileIndex}*inv(T_ds_hom^(scaleIndex-1)) ;
        else
            T_in{fileIndex} = inv(T_ds_hom)*T_in{fileIndex}*T_ds_hom ;
        end
        
        % for display purposes 与展示相关的操作
        if raslpara.DISPLAY > 0
            fr = [1 1          imgSize(2) imgSize(2) 1; ...
                  1 imgSize(1) imgSize(1) 1          1; ...
                  1 1          1          1          1 ];
            
            % 输入的初始变换
            frOrig{fileIndex} = T_in{fileIndex} * fr;
        end
    end
    
    % 开始迭代，直到收敛
    while ~converged
        
        % numIterOuter指的是所有尺度外层循环的总的次数，而iterNum是某个尺度上外层循环的迭代次数
        % 对于尺度是1的情况，当然它们是一样的
        iterNum = iterNum + 1 ;
        numIterOuter = numIterOuter + 1 ;
        
        %D = [] ;
        D= zeros(imgSize(1)*imgSize(2), numImages); % 注意一下这里的D和上面的D不一样了
        J = cell(1,numImages) ;
        disp(['Scale ' num2str(scaleIndex) '  Iter ' num2str(iterNum)]) ;
        
        % 这个循环的作用就是求解D和J
        % D是输入影像按照初始变换参数变换后的影像，J是变换后的影像对应的雅可比矩阵
        for fileIndex = 1 : numImages

            % transformed image and derivatives with respect to affine parameters
            % 将影像与对应x、y方向的梯度影像应用变换进行重采
            % 这里的T_in就是上面赋值得到的，而且在尺度为1的情况下T_in就等于输入的初始transformations
            % 构造变换矩阵
            Tfm = fliptform(maketform('projective',T_in{fileIndex}'));
            
            % 根据每个影像对应的变换矩阵Tfm利用imtransform函数对输入影像进行变换重采，重采插值方法时bicubic-双三次插值
            I   = vec(imtransform(I0{scaleIndex,fileIndex}, Tfm,'bicubic','XData',[1 imgSize(2)],'YData',[1 imgSize(1)],'Size',imgSize));
            Iu  = vec(imtransform(I0x{scaleIndex,fileIndex},Tfm,'bicubic','XData',[1 imgSize(2)],'YData',[1 imgSize(1)],'Size',imgSize));
            Iv  = vec(imtransform(I0y{scaleIndex,fileIndex},Tfm,'bicubic','XData',[1 imgSize(2)],'YData',[1 imgSize(1)],'Size',imgSize));
            y   = I; %vec(I);

            Iu = (1/norm(y))*Iu - ( (y'*Iu)/(norm(y))^3 )*y ;   % 对x梯度影像大小进行归一化
            Iv = (1/norm(y))*Iv - ( (y'*Iv)/(norm(y))^3 )*y ;   % 对y梯度影像大小进行归一化

            y = y / norm(y) ; % normalize 对影像进行灰度归一化
            % D = [D y] ;
            % 需要注意的是，后续迭代的输入都是基于这个按初始变换后的影像进行的
            % 换句话说，如果初始变换给错了，直接会导致这里的D不对，进而导致后续迭代结果出错
            D(:,fileIndex) = y ; % 将变换过的、灰度归一化好的影像追加到D中

            % transformation matrix to parameters 将变换矩阵转成变换参数，和上面是一样的
            xi{fileIndex} = projective_matrix_to_parameters(raslpara.transformType,T_in{fileIndex}) ; 
            
            % Compute Jacobian 计算每张影像对应的雅可比矩阵
            J{fileIndex} = image_Jaco(Iu, Iv, imgSize, raslpara.transformType, xi{fileIndex});
        end
        
        % 计算目标函数中的lambda，对应论文中公式5下面的内容
        lambda = raslpara.lambdac/sqrt(size(D,1)) ; 

        
        % RASL inner loop RASL的内层循环
        % -----------------------------------------------------------------
        % -----------------------------------------------------------------
        % using QR to orthogonalize the Jacobian matrix 使用QR分解来正交化雅可比矩阵
        for fileIndex = 1 : numImages
            [Q{fileIndex}, R{fileIndex}] = qr(J{fileIndex},0) ;
        end
        
        % 提供了两种内层循环的迭代算法：APG是1，ALM是0
        % 内层循环会返回低秩部分A、稀疏部分E、初始变换的修正delta_xi
        if APGorALM_flag == 1
            % D：上面构造的按照初始变换参数变换后的影像
            % Q：QR分解后得到的Q
            % lambda：上面按照公式计算出来的数值
            % raslpara.inner_tol：传入的参数，内层循环的迭代终止条件
            % raslpara.inner_maxIter：传入的参数，内层循环的最大迭代次数
            % raslpara.continuationFlag：传入的参数，内层循环是否继续的flag参数
            % raslpara.mu：relaxation parameter
            [A, E, delta_xi, numIterInnerEach] = rasl_inner_apg(D, Q, lambda, raslpara.inner_tol, raslpara.inner_maxIter, raslpara.continuationFlag, raslpara.mu) ;
        else
            [A, E, delta_xi, numIterInnerEach] = rasl_inner_ialm(D, Q, lambda, raslpara.inner_tol, raslpara.inner_maxIter);
        end
        
        % 对于每个影像得到最终的初始变换修正delta_xi
        for fileIndex = 1 : numImages
            delta_xi{fileIndex} = inv(R{fileIndex})*delta_xi{fileIndex} ;
        end
        % -----------------------------------------------------------------
        % -----------------------------------------------------------------
        
        % 内层循环总的迭代次数累计
        numIterInner = numIterInner + numIterInnerEach ;
        
        % 计算当前目标函数的值，对应论文中的公式5
        curObj = norm(svd(A),1) + lambda*norm(E(:),1) ;
        disp(['previous objective function: ' num2str(prevObj) ]);
        disp([' current objective function: ' num2str(curObj) ]);

        % step in paramters 更新输入的初始变换参数(矩阵)
        % 到这一步的时候，xi和T_in就是最新的变换参数、矩阵了
        for i = 1 : numImages
            xi{i} = xi{i} + delta_xi{i};
            T_in{i} = parameters_to_projective_matrix(raslpara.transformType,xi{i});
        end
        
        % save intermedia results 是否保存每次外层迭代的中间结果，默认是否
        if raslpara.saveIntermedia
            matName = strcat('scale_', num2str(scaleIndex),'_iter_', num2str(iterNum),'.mat') ;
            save(fullfile(destDir, matName),'D','A','E','xi') ;
        end
        
        % 如果展示每次外层迭代结果，执行下面代码
        if raslpara.DISPLAY > 0
            % 对于每个影像，依次展示
            for i = 1 : numImages
                figure(1); clf ;
                % 显示未经任何处理的影像作为底图
                imshow(I0{scaleIndex,i},[],'Border','tight');
                hold on;
                
                % 根据最新的变换构建变换矩阵
                Tfm = fliptform(maketform('projective',inv(T_in{i}')));
                % curFrame对应迭代更新后的变换
                curFrame = tformfwd(fr(1:2,:)', Tfm )';
                % 迭代更新后的变换用红色画出，原来的变换用绿色画出
                plot( frOrig{i}(1,:),   frOrig{i}(2,:),   'g-', 'LineWidth', 2 );
                plot( curFrame(1,:), curFrame(2,:), 'r-', 'LineWidth', 2 );
%                 hold off;
%                 print('-f1', '-dbmp', fullfile(destDir, num2str(i))) ;
            end
        end
        
        % 如果上一次的迭代目标值与当前值小于终止条件或者迭代次数达到了最大迭代次数就停止
        if ( (prevObj - curObj < raslpara.stoppingDelta) || iterNum >= raslpara.maxIter )
            % 强行认为收敛了
            converged = 1;
            % 如果是达到了最大迭代次数，最后简单输出说明一下
            if ( prevObj - curObj >= raslpara.stoppingDelta )
                disp('Maximum iterations reached') ;
            end
        else
            % 如果没有小于终止条件就将当前值赋给prevObj，然后执行下一次循环
            prevObj = curObj;
        end

    end
end

timeConsumed = toc

disp(['total number of iterations: ' num2str(numIterInner) ]);
disp(['number of outer loop: ' num2str(numIterOuter) ]);

%% save the alignment results 保存对齐结果

Do = [] ;
for fileIndex = 1 : numImages
    % 这里的Tfm就是最后迭代得到的变换矩阵结果
    % maketform用于生成了一个投影变换,flip函数用于获得反向的变换
    % 前面说过了T_in是更新后的初始变换，所以这里就直接拿来用了
    Tfm = fliptform(maketform('projective',T_in{fileIndex}'));
    
    % 这一行其实就是对影像按照上面得到的变换进行重采
    % 重采的插值算法是bicubic-双三次插值
    I   = vec(imtransform(I0{1,fileIndex}, Tfm,'bicubic','XData',[1 imgSize(2)],'YData',[1 imgSize(1)],'Size',imgSize));
    y   = I; 
    % 在重采之后，对影像的灰度值做了归一化处理
    y = y / norm(y) ; % normalize
    
    % 最后将得到的对齐的影像append到Do的后面
    Do = [Do y] ;
end

% 如果需要保存结果，就把得到的Do、A、E、xi都保存到final.mat里
% 这里一定要注意保存的对齐的影像的灰度是经过归一化处理后的，并不是8bit量化或是其它什么
if raslpara.saveEnd
    save(fullfile(destDir, 'final.mat'),'Do','A','E','xi') ;
end

% 保存一些基本的运行信息
outputFileName = fullfile(destDir, 'results.txt'); 
fid = fopen(outputFileName,'a') ;
fprintf(fid, '%s\n', [' total number of iterations: ' num2str(numIterInner) ]) ;
fprintf(fid, '%s\n', [' number of outer loop ' num2str(numIterOuter) ]) ;
fprintf(fid, '%s\n', [' consumed time: ' num2str(timeConsumed)]) ;
fprintf(fid, '%s\n', [' the parameters :']) ;
fprintf(fid, '%s\n', [' transformType ' raslpara.transformType ]) ;
fprintf(fid, '%s\n', [' lambda ' num2str(raslpara.lambdac) ' times sqrt(m)']) ;
fprintf(fid, '%s\n', [' stoppingDelta of outer loop ' num2str(raslpara.stoppingDelta) ]) ;
fprintf(fid, '%s\n', [' stoppingDelta of inner loop ' num2str(raslpara.inner_tol)]) ;
if APGorALM_flag == 1
    fprintf(fid, '%s\n', [' optimization in inner loop is using APG algorithm']) ;
    fprintf(fid, '%s\n', [' continuationFlag  ' num2str(raslpara.continuationFlag) ]) ;
    fprintf(fid, '%s\n', [' mu of inner loop ' num2str(raslpara.mu) ]) ;
else
    fprintf(fid, '%s\n', [' optimization in inner loop is using inexact ALM algorithm']) ;
end
fclose(fid);

