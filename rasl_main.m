function [D, Do, A, E, xi, numIterOuter, numIterInner ] = rasl_main(fileNames, transformations, numImages, raslpara, destDir)

% ---------------------------------------------------
% Batch image alignment: RASL main loop
%
% input: fileNames               --- the names of images Ӱ�������
%        transformations         --- the initial transformation matrix ��ʼ�任����
%        numImages               --- the number of images Ӱ��ĸ���
%        raslpara                --- parameters for RASL RASL����
%        destDir                 --- output ���·��
%
% output: D                      --- input images in canonical frame ԭʼ����ϵ�е�Ӱ��
%         Do                     --- output aligned images ����Ķ���Ӱ��
%         A                      --- low-rank component ���Ȳ���
%         E                      --- sparse error component ϡ�貿��
%         xi                     --- transformation parameters �任����
%         numIterOuter           --- number of outer loop iterations ����������
%         numIterInner           --- total number of inner loop iterations �ڲ��������
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

% cell(m,n)�������ڽ���һ��m��n�Ŀյ�����(����)
I0 = cell(raslpara.numScales,numImages) ; % images �����û�о����κδ����Ӱ��
I0x = cell(raslpara.numScales,numImages) ; % image derivatives �����ԭʼӰ���Ӧ��x��y������ݶ�Ӱ��
I0y = cell(raslpara.numScales,numImages) ;

for fileIndex = 1 : numImages
    % ��double��ʽ��ȡ��ǰ�ļ�������Ӧ��Ӱ��
    currentImage = double(imread(fileNames{fileIndex}));
    
    % �����size������һ�������Ƕ��󣬵ڶ��������Ƕ����ά��
    % ����A��һ��4��5�ľ���size(A,1)�ͱ�ʾA�ĵ�һά�ĳ��ȣ�Ҳ����4
    % Use only the green channel in case of color images    
    if size(currentImage,3) > 1,   currentImage = currentImage(:,:,2);            end
    if deGammaTraining,      currentImage = gamma_decompress(currentImage, gammaType); end
    
    % ������˹������
    currentImagePyramid = gauss_pyramid( currentImage, raslpara.numScales,...
        sqrt(det(transformations{fileIndex}(1:2,1:2)))*sigma0, sigmaS );
    
    % ����һ��forѭ��������������ͬ�߶ȣ���numScales��ʼ��1����������Ϊ-1
    for scaleIndex = raslpara.numScales:-1:1
        I0{scaleIndex,fileIndex} = currentImagePyramid{scaleIndex};
        
        % image derivatives ��Ӱ����
        % fspecial�������ڴ���һ��Ԥ������˲�����
        % imfilter�������ڶ�Ӱ��ָ�������˲�
        I0_smooth = I0{scaleIndex,fileIndex};
        I0x{scaleIndex,fileIndex} = imfilter( I0_smooth, (-fspecial('sobel')') / 8 );
        I0y{scaleIndex,fileIndex} = imfilter( I0_smooth,  -fspecial('sobel')   / 8 );
    end
end



%% get the initial input images in canonical frame

imgSize = raslpara.canonicalImageSize ; 

xi_initial = cell(1,numImages) ; % initial transformation parameters ���ڴ�ų�ʼ�任����
for i = 1 : numImages
    % ������ı任��������жϣ�����任���������С��3�����������׷��һ��0 0 1
    % ��get_training_images���������IDENTITY���ص���һ��3��3�ĵ�λ��SIMILARITY��AFFINE���صĶ���2��3�ľ���
    if size(transformations{i},1) < 3
        transformations{i} = [transformations{i} ; 0 0 1] ;
    end
    % ������һ���Ժ󣬵õ��ľ���һ����׼��3��3�Ķ�άӰ��ı任�����ˣ��ٽ���ת��Ϊ�任����
    % ����RASL���ԣ�ֻ֧���ض��ļ��ֱ任������TRANSLATION��AFFINE��SIMILARITY��HOMOGRAPHY��EUCLIDEAN�������������;���Ϊ��δʶ�����
    % ��ʵ�������⼸�ֱ任��������Ķ�ά�任�Ѿ��㹻�ˣ������ڿ��ܴ��ھֲ������ң��Ӱ����ԣ�ͬ��Ҳ������Ϊ��
    % ���ڲ�ͬ���͵ı任��xi�Ĳ�������Ҳ��ͬ��
    % TRANSLATION 2��������dx��dy
    % EUCLIDEAN   3����������ת�Ƕ�(����)��xƽ�Ʒ�����yƽ�Ʒ���
    % SIMILARITY  4�����������Ų�������ת�Ƕ�(����)��xƽ�Ʒ�����yƽ�Ʒ���
    % AFFINE      6��������r11��r12��dx��r21��r22��dy
    % HOMOGRAPHY  8��������h11��h12��h13��h21��h22��h23��h31��h32
    xi_initial{i} = projective_matrix_to_parameters(raslpara.transformType,transformations{i});
end

%D = [] ;
D= zeros(imgSize(1)*imgSize(2), numImages);
for fileIndex = 1 : numImages
    % transformed image 
    Tfm = fliptform(maketform('projective',transformations{fileIndex}'));
    
    % ��Ӱ���ճ�ʼ�任���б任���ز��ڲ巽ʽ��bicubic-˫���β�ֵ
    I   = vec(imtransform(I0{1,fileIndex}, Tfm,'bicubic','XData',[1 imgSize(2)],'YData',[1 imgSize(1)],'Size',imgSize));
    y   = I; 
    y = y / norm(y) ; % normalize ��Ӱ��ĻҶȹ�һ��

    %D = [D y] ;
    D(:,fileIndex) = y ;    % �����ճ�ʼ�任��ġ��Ҷȹ�һ����Ӱ��׷�ӵ�D��
end

% ��������˱����ʼ״̬���ͽ�D(���ճ�ʼ�任��ġ��Ҷȹ�һ����Ӱ��)��xi_initial(��ʼ�任����)���浽original.mat
% ��Ҫע����ǣ����ﱣ����ǰ��ճ�ʼ�任����任����Ӱ�񣬶�����ֱ�Ӷ�ȡ��Ӱ��
if raslpara.saveStart
    save(fullfile(destDir, 'original.mat'),'D','xi_initial');
end

% ��������Ĳ�����ʵ����RASL���ڽ���ϴ���ȵ�misalignment�İ취
% ����һ����ʼ�任������ϴ��Ӱ����任�زɵ��������״̬�����ڴ�״̬�ٽ��е���
% ֮����˵�������״̬����Ϊ����ѡ��Ķ�Ӧ��Թ����ĳ�ʼ״̬��һ���ϸ񡢾�ȷ


%% start the main loop ��ʼRASL����ѭ��

frOrig = cell(1,numImages) ;
T_in = cell(1,numImages) ;

T_ds = [ 0.5,   0, -0.5; ...
         0,   0.5, -0.5   ];
T_ds_hom = [ T_ds; [ 0 0 1 ]];

numIterOuter = 0 ;  % ���ѭ���ܵĵ�������
numIterInner = 0 ;  % �ڲ�ѭ���ܵĵ�������

tic % time counting start

for scaleIndex = raslpara.numScales:-1:1 % multiscale ���ڽ����߶����⣬����ĳ߶���Ϊ1������ִֻ��1��
    
    iterNum = 0 ;  % iteration number of outer loop in each scale ÿ���߶ȵ����ѭ������
    converged = 0 ; % flag�����������Ƿ�����
    prevObj = inf ; % previous objective function value ��һ�ε�����Ŀ�꺯��ֵ
    
    % ���ڳ߶���1����������imgSize�͵���raslpara.canonicalImageSize
    imgSize = raslpara.canonicalImageSize / 2^(scaleIndex-1) ;    
    xi = cell(1,numImages) ;
    
    % ���ڿ��ǵ��߶����⣬�Գ�ʼ�任������д�����ÿ���任����ת��һ�±��T_in
    for fileIndex = 1 : numImages
        % scaleIndexΪ1��raslpara.numScalesҲΪ1������ִ��if���
        if scaleIndex == raslpara.numScales
            % ���ڲ�û���漰���߶����⣬scaleIndexΪ1�����������T_in������ĳ�ʼtransformations��ͬ
            T_in{fileIndex} = T_ds_hom^(scaleIndex-1)*transformations{fileIndex}*inv(T_ds_hom^(scaleIndex-1)) ;
        else
            T_in{fileIndex} = inv(T_ds_hom)*T_in{fileIndex}*T_ds_hom ;
        end
        
        % for display purposes ��չʾ��صĲ���
        if raslpara.DISPLAY > 0
            fr = [1 1          imgSize(2) imgSize(2) 1; ...
                  1 imgSize(1) imgSize(1) 1          1; ...
                  1 1          1          1          1 ];
            
            % ����ĳ�ʼ�任
            frOrig{fileIndex} = T_in{fileIndex} * fr;
        end
    end
    
    % ��ʼ������ֱ������
    while ~converged
        
        % numIterOuterָ�������г߶����ѭ�����ܵĴ�������iterNum��ĳ���߶������ѭ���ĵ�������
        % ���ڳ߶���1���������Ȼ������һ����
        iterNum = iterNum + 1 ;
        numIterOuter = numIterOuter + 1 ;
        
        %D = [] ;
        D= zeros(imgSize(1)*imgSize(2), numImages); % ע��һ�������D�������D��һ����
        J = cell(1,numImages) ;
        disp(['Scale ' num2str(scaleIndex) '  Iter ' num2str(iterNum)]) ;
        
        % ���ѭ�������þ������D��J
        % D������Ӱ���ճ�ʼ�任�����任���Ӱ��J�Ǳ任���Ӱ���Ӧ���ſɱȾ���
        for fileIndex = 1 : numImages

            % transformed image and derivatives with respect to affine parameters
            % ��Ӱ�����Ӧx��y������ݶ�Ӱ��Ӧ�ñ任�����ز�
            % �����T_in�������渳ֵ�õ��ģ������ڳ߶�Ϊ1�������T_in�͵�������ĳ�ʼtransformations
            % ����任����
            Tfm = fliptform(maketform('projective',T_in{fileIndex}'));
            
            % ����ÿ��Ӱ���Ӧ�ı任����Tfm����imtransform����������Ӱ����б任�زɣ��زɲ�ֵ����ʱbicubic-˫���β�ֵ
            I   = vec(imtransform(I0{scaleIndex,fileIndex}, Tfm,'bicubic','XData',[1 imgSize(2)],'YData',[1 imgSize(1)],'Size',imgSize));
            Iu  = vec(imtransform(I0x{scaleIndex,fileIndex},Tfm,'bicubic','XData',[1 imgSize(2)],'YData',[1 imgSize(1)],'Size',imgSize));
            Iv  = vec(imtransform(I0y{scaleIndex,fileIndex},Tfm,'bicubic','XData',[1 imgSize(2)],'YData',[1 imgSize(1)],'Size',imgSize));
            y   = I; %vec(I);

            Iu = (1/norm(y))*Iu - ( (y'*Iu)/(norm(y))^3 )*y ;   % ��x�ݶ�Ӱ���С���й�һ��
            Iv = (1/norm(y))*Iv - ( (y'*Iv)/(norm(y))^3 )*y ;   % ��y�ݶ�Ӱ���С���й�һ��

            y = y / norm(y) ; % normalize ��Ӱ����лҶȹ�һ��
            % D = [D y] ;
            % ��Ҫע����ǣ��������������붼�ǻ����������ʼ�任���Ӱ����е�
            % ���仰˵�������ʼ�任�����ˣ�ֱ�ӻᵼ�������D���ԣ��������º��������������
            D(:,fileIndex) = y ; % ���任���ġ��Ҷȹ�һ���õ�Ӱ��׷�ӵ�D��

            % transformation matrix to parameters ���任����ת�ɱ任��������������һ����
            xi{fileIndex} = projective_matrix_to_parameters(raslpara.transformType,T_in{fileIndex}) ; 
            
            % Compute Jacobian ����ÿ��Ӱ���Ӧ���ſɱȾ���
            J{fileIndex} = image_Jaco(Iu, Iv, imgSize, raslpara.transformType, xi{fileIndex});
        end
        
        % ����Ŀ�꺯���е�lambda����Ӧ�����й�ʽ5���������
        lambda = raslpara.lambdac/sqrt(size(D,1)) ; 

        
        % RASL inner loop RASL���ڲ�ѭ��
        % -----------------------------------------------------------------
        % -----------------------------------------------------------------
        % using QR to orthogonalize the Jacobian matrix ʹ��QR�ֽ����������ſɱȾ���
        for fileIndex = 1 : numImages
            [Q{fileIndex}, R{fileIndex}] = qr(J{fileIndex},0) ;
        end
        
        % �ṩ�������ڲ�ѭ���ĵ����㷨��APG��1��ALM��0
        % �ڲ�ѭ���᷵�ص��Ȳ���A��ϡ�貿��E����ʼ�任������delta_xi
        if APGorALM_flag == 1
            % D�����湹��İ��ճ�ʼ�任�����任���Ӱ��
            % Q��QR�ֽ��õ���Q
            % lambda�����水�չ�ʽ�����������ֵ
            % raslpara.inner_tol������Ĳ������ڲ�ѭ���ĵ�����ֹ����
            % raslpara.inner_maxIter������Ĳ������ڲ�ѭ��������������
            % raslpara.continuationFlag������Ĳ������ڲ�ѭ���Ƿ������flag����
            % raslpara.mu��relaxation parameter
            [A, E, delta_xi, numIterInnerEach] = rasl_inner_apg(D, Q, lambda, raslpara.inner_tol, raslpara.inner_maxIter, raslpara.continuationFlag, raslpara.mu) ;
        else
            [A, E, delta_xi, numIterInnerEach] = rasl_inner_ialm(D, Q, lambda, raslpara.inner_tol, raslpara.inner_maxIter);
        end
        
        % ����ÿ��Ӱ��õ����յĳ�ʼ�任����delta_xi
        for fileIndex = 1 : numImages
            delta_xi{fileIndex} = inv(R{fileIndex})*delta_xi{fileIndex} ;
        end
        % -----------------------------------------------------------------
        % -----------------------------------------------------------------
        
        % �ڲ�ѭ���ܵĵ��������ۼ�
        numIterInner = numIterInner + numIterInnerEach ;
        
        % ���㵱ǰĿ�꺯����ֵ����Ӧ�����еĹ�ʽ5
        curObj = norm(svd(A),1) + lambda*norm(E(:),1) ;
        disp(['previous objective function: ' num2str(prevObj) ]);
        disp([' current objective function: ' num2str(curObj) ]);

        % step in paramters ��������ĳ�ʼ�任����(����)
        % ����һ����ʱ��xi��T_in�������µı任������������
        for i = 1 : numImages
            xi{i} = xi{i} + delta_xi{i};
            T_in{i} = parameters_to_projective_matrix(raslpara.transformType,xi{i});
        end
        
        % save intermedia results �Ƿ񱣴�ÿ�����������м�����Ĭ���Ƿ�
        if raslpara.saveIntermedia
            matName = strcat('scale_', num2str(scaleIndex),'_iter_', num2str(iterNum),'.mat') ;
            save(fullfile(destDir, matName),'D','A','E','xi') ;
        end
        
        % ���չʾÿ�������������ִ���������
        if raslpara.DISPLAY > 0
            % ����ÿ��Ӱ������չʾ
            for i = 1 : numImages
                figure(1); clf ;
                % ��ʾδ���κδ����Ӱ����Ϊ��ͼ
                imshow(I0{scaleIndex,i},[],'Border','tight');
                hold on;
                
                % �������µı任�����任����
                Tfm = fliptform(maketform('projective',inv(T_in{i}')));
                % curFrame��Ӧ�������º�ı任
                curFrame = tformfwd(fr(1:2,:)', Tfm )';
                % �������º�ı任�ú�ɫ������ԭ���ı任����ɫ����
                plot( frOrig{i}(1,:),   frOrig{i}(2,:),   'g-', 'LineWidth', 2 );
                plot( curFrame(1,:), curFrame(2,:), 'r-', 'LineWidth', 2 );
%                 hold off;
%                 print('-f1', '-dbmp', fullfile(destDir, num2str(i))) ;
            end
        end
        
        % �����һ�εĵ���Ŀ��ֵ�뵱ǰֵС����ֹ�������ߵ��������ﵽ��������������ֹͣ
        if ( (prevObj - curObj < raslpara.stoppingDelta) || iterNum >= raslpara.maxIter )
            % ǿ����Ϊ������
            converged = 1;
            % ����Ǵﵽ���������������������˵��һ��
            if ( prevObj - curObj >= raslpara.stoppingDelta )
                disp('Maximum iterations reached') ;
            end
        else
            % ���û��С����ֹ�����ͽ���ǰֵ����prevObj��Ȼ��ִ����һ��ѭ��
            prevObj = curObj;
        end

    end
end

timeConsumed = toc

disp(['total number of iterations: ' num2str(numIterInner) ]);
disp(['number of outer loop: ' num2str(numIterOuter) ]);

%% save the alignment results ���������

Do = [] ;
for fileIndex = 1 : numImages
    % �����Tfm�����������õ��ı任������
    % maketform����������һ��ͶӰ�任,flip�������ڻ�÷���ı任
    % ǰ��˵����T_in�Ǹ��º�ĳ�ʼ�任�����������ֱ����������
    Tfm = fliptform(maketform('projective',T_in{fileIndex}'));
    
    % ��һ����ʵ���Ƕ�Ӱ��������õ��ı任�����ز�
    % �زɵĲ�ֵ�㷨��bicubic-˫���β�ֵ
    I   = vec(imtransform(I0{1,fileIndex}, Tfm,'bicubic','XData',[1 imgSize(2)],'YData',[1 imgSize(1)],'Size',imgSize));
    y   = I; 
    % ���ز�֮�󣬶�Ӱ��ĻҶ�ֵ���˹�һ������
    y = y / norm(y) ; % normalize
    
    % ��󽫵õ��Ķ����Ӱ��append��Do�ĺ���
    Do = [Do y] ;
end

% �����Ҫ���������Ͱѵõ���Do��A��E��xi�����浽final.mat��
% ����һ��Ҫע�Ᵽ��Ķ����Ӱ��ĻҶ��Ǿ�����һ�������ģ�������8bit������������ʲô
if raslpara.saveEnd
    save(fullfile(destDir, 'final.mat'),'Do','A','E','xi') ;
end

% ����һЩ������������Ϣ
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

