function [options, nmfStructNames] = getRoisNMF(D, meta, plotoutputs)

%clear;
%% load file

addpath(genpath('utilities'));
             
% source_dir = '/nas/volume1/2photon/RESDATA/20161221_JR030W/retinotopy037Hz';
% acquisitionName = 'fov1_bar037Hz_run4';
source_dir = D.sourceDir;
acquisitionName = D.acquisitionName;

maskPath = fullfile(D.datastructPath, 'masks');
if ~exist(maskPath, 'dir')
    mkdir(maskPath);
end

tiffPaths = meta.tiffPaths;
slicesToUse = D.slices;
nTiffs = meta.nTiffs;
nSlices = length(slicesToUse);

%%
scaleFOV = D.maskInfo.params.scaleFOV;
removeBadFrames = D.maskInfo.params.removeBadFrames;

%% Load sample movie for params:


currSlice = 1;
currFile = 1;
currTiffs = dir(fullfile(tiffPaths{currFile}, '*.tif'))';
currTiffs = {currTiffs(:).name}';

% tif_path = fullfile(source_dir, sprintf('Corrected_Channel01_File%03d_scaled', currFile));
% tif_name = sprintf('%s_Slice%02d_Channel01_File%03d.tif', acquisitionName, currSlice, currFile);

tif_path = tiffPaths{currFile};
tif_name = currTiffs{currSlice};
nam = fullfile(tif_path, tif_name);

sframe=1;						% user input: first frame to read (optional, default 1)
Y = bigread2(nam,sframe);

if ~isa(Y,'double');    Y = double(Y);  end         % convert to single


% -------------------------------------------------------------------------
% ISSUES:
% -------------------------------------------------------------------------
% 1. Scale FOV or no? Current params fail to find good ROIs when squashed.
% -------------------------------------------------------------------------
% -- if don't rescale first, can always scale ROI patches later...

if scaleFOV
    Y = imresize(Y, [size(Y,1)*2, size(Y,2)]);
end

% -------------------------------------------------------------------------
% 2. Replace "bad" motion-corrected frames or no?
% -------------------------------------------------------------------------
% -- The following will replace bad frames w/ NaNs... How does this perform
% with current NMF pipeline?

if removeBadFrames
    checkframes = @(x,y) corrcoef(x,y);
    refframe = 1;
    corrs = arrayfun(@(i) checkframes(Y(:,:,i), Y(:,:,refframe)), 1:size(Y,3), 'UniformOutput', false);
    corrs = cat(3, corrs{1:end});
    meancorrs = squeeze(mean(mean(corrs,1),2));
    badframes = find(abs(meancorrs-mean(meancorrs))>=std(meancorrs)*3); %find(meancorrs<0.795);

    if length(badframes)>1
        fprintf('Bad frames found in movie %s at: %s\n', currSliceName, mat2str(badframes(2:end)));
    end
    while length(badframes) >= size(Y,3)*0.25
        refframe = refframe +1; 
        corrs = arrayfun(@(i) checkframes(Y(:,:,i), Y(:,:,1)), 1:size(Y,3), 'UniformOutput', false);
        corrs = cat(3, corrs{1:end});
        meancorrs = squeeze(mean(mean(corrs,1),2));
        badframes = find(abs(meancorrs-mean(meancorrs))>=std(meancorrs)*2); %find(meancorrs<0.795);
    end
end
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------


[d1,d2,T] = size(Y);                                % dimensions of dataset
d = d1*d2; 

%% Set parameters

% K = 500; %50; %300; %50; %150; %35;                                      % number of components to be found
% tau = 2; %2; %4;                                      % std of gaussian kernel (size of neuron) 
% 
% % tau = [1 1; 2 2];
% % K = [100; 50];
% 
% p = 2;     % order of autoregressive system (p = 0 no dynamics, p=1 just decay, p = 2, both rise and decay)
% merge_thr = 0.8;                                  % merging threshold
% 
%options = D.params.options;

K = D.params.K;
tau = D.params.tau;
merge_thr = D.params.merge_thr;
p = D.params.p;

options = CNMFSetParms(...                      
    'd1',d1,'d2',d2,...                         % dimensions of datasets
    'search_method','dilate','dist',3,...       % search locations when updating spatial components
    'deconv_method','constrained_foopsi',...    % activity deconvolution method
    'temporal_iter',2,...                       % number of block-coordinate descent steps 
    'fudge_factor',0.98,...                     % bias correction for AR coefficients
    'merge_thr',merge_thr,...                    % merging threshold
    'gSig',tau,...%'method','lars',... %jyr
    'thr_method', 'nrg'... %'max'...
    );


%% Go through each slice and save ROIs to struct:

nmfStructNames = {};

for sliceidx=1:nSlices
    currSlice = slicesToUse(sliceidx);

    fprintf('=========================================================\n');
    fprintf('Running NMF on Slice %03d.\n', currSlice);
    fprintf('=========================================================\n');
    
    nmfstruct = struct();
    
    for tiffidx=1:nTiffs
        currFile = tiffidx;

        currTiffs = dir(fullfile(tiffPaths{currFile}, '*.tif'))';
        currTiffs = {currTiffs(:).name}';

        tif_path = tiffPaths{currFile};
        tif_name = currTiffs{currSlice};
        nam = fullfile(tif_path, tif_name);

        fprintf('Starting: Slice%02d, File%03d...\n', currSlice, currFile);
        fprintf('TIFF name is %s.\n', tif_name);


        nam = fullfile(tif_path, tif_name);

        sframe=1;						% user input: first frame to read (optional, default 1)
        %num2read=2000;					% user input: how many frames to read   (optional, default until the end)
        Y = bigread2(nam,sframe);

        if ~isa(Y,'double');    Y = double(Y);  end         % convert to single



        if scaleFOV
            fprintf('Rescaling...\n');
            Y = imresize(Y, [size(Y,1)*2, size(Y,2)]);
        end

        % -------------------------------------------------------------------------
        % 2. Replace "bad" motion-corrected frames or no?
        % -------------------------------------------------------------------------
        % -- The following will replace bad frames w/ NaNs... How does this perform
        % with current NMF pipeline?

        if removeBadFrames
            fprintf('Removing bad frames...\n');
            checkframes = @(x,y) corrcoef(x,y);
            refframe = 1;
            corrs = arrayfun(@(i) checkframes(Y(:,:,i), Y(:,:,refframe)), 1:size(Y,3), 'UniformOutput', false);
            corrs = cat(3, corrs{1:end});
            meancorrs = squeeze(mean(mean(corrs,1),2));
            badframes = find(abs(meancorrs-mean(meancorrs))>=std(meancorrs)*3); %find(meancorrs<0.795);

            if length(badframes)>1
                fprintf('Bad frames found in movie %s at: %s\n', currSliceName, mat2str(badframes(2:end)));
            end
            while length(badframes) >= size(Y,3)*0.25
                refframe = refframe +1; 
                corrs = arrayfun(@(i) checkframes(Y(:,:,i), Y(:,:,1)), 1:size(Y,3), 'UniformOutput', false);
                corrs = cat(3, corrs{1:end});
                meancorrs = squeeze(mean(mean(corrs,1),2));
                badframes = find(abs(meancorrs-mean(meancorrs))>=std(meancorrs)*2); %find(meancorrs<0.795);
            end
        end



        [d1,d2,T] = size(Y);                                % dimensions of dataset
        d = d1*d2;                                          % total number of pixels

        %% Data pre-processing

        % High-pass filter:
        % winsize = 25;
        % for i=119:d1
        %    for j=396:d2
        %        tcurr = Y(i,j,:);
        %        s1 = smooth(squeeze(tcurr), winsize, 'rlowess');
        %        t1 = squeeze(tcurr) - s1;
        %        Y(i,j,:) = t1;
        %    end
        % end
        % 
        % frameArray = reshape(Y, d, size(Y,3));
        % framesToAvg = 5;
        % for pix = 1:length(frameArray)
        %     tmp0=frameArray(pix,:);
        %     tmp1=padarray(tmp0,[0 framesToAvg],tmp0(1),'pre');
        %     tmp1=padarray(tmp1,[0 framesToAvg],tmp0(end),'post');
        %     rollingAvg=conv(tmp1,fspecial('average',[1 framesToAvg]),'same');%average
        %     rollingAvg=rollingAvg(framesToAvg+1:end-framesToAvg);
        %     frameArray(pix,:) = tmp0 - rollingAvg;
        % end
        % 
        % save(strcat(tif_path, sprintf('Y_rollavg_%i.mat', framesToAvg)), 'frameArray');
        % 
        % figure(); plot(tmp0,'k'); hold on;
        % plot(rollingAvg, 'r');
        % hold on;
        % plot(tmp0-rollingAvg, 'g');


        % 
        % figure();
        % plot(t1, 'k')
        % hold on;
        % plot(squeeze(tcurr), 'k')
        % 
        % figure()
        % d1 = (t1 - mean(t1)) / mean(t1);
        % plot(d1, 'k');
        % hold on;
        % 

        % save(strcat(tif_path,'Y_highpas.mat'), 'Y');

        %% fast initialization of spatial components using greedyROI and HALS

        [P,Y] = preprocess_data(Y,p);


        [Ain,Cin,bin,fin,center] = initialize_components(Y,K,tau,options,P);  % initialize

        % display centers of found components
        Cn =  correlation_image(Y); %reshape(P.sn,d1,d2);  %max(Y,[],3); %std(Y,[],3); % image statistic (only for display purposes)
        if plotoutputs
            figure;imagesc(Cn);
                axis equal; axis tight; hold all;
                scatter(center(:,2),center(:,1),'mo');
                title('Center of ROIs found from initialization algorithm');
                drawnow;
        end

        %% manually refine components (optional)
        refine_components = false;  % flag for manual refinement
        if refine_components
            [Ain,Cin,center] = manually_refine_components(Y,Ain,Cin,center,Cn,tau,options);
        end

        %% update spatial components
        Yr = reshape(Y,d,T);
        [A,b,Cin] = update_spatial_components(Yr,Cin,fin,[Ain,bin],P,options);

        %% update temporal components
        P.p = 0;    % set AR temporarily to zero for speed
        [C,f,P,S,YrA] = update_temporal_components(Yr,A,b,Cin,fin,P,options);

        %% merge found components
        [Am,Cm,K_m,merged_ROIs,Pm,Sm] = merge_components(Yr,A,b,C,f,P,S,options);

        %%
        if plotoutputs
            display_merging = 1; % flag for displaying merging example
        else
            display_merging = 0;
        end
        if and(display_merging, ~isempty(merged_ROIs))
            i = 1; %randi(length(merged_ROIs));
            ln = length(merged_ROIs{i});
            if plotoutputs
                figure;
                    set(gcf,'Position',[300,300,(ln+2)*300,300]);
                    for j = 1:ln
                        subplot(1,ln+2,j); imagesc(reshape(A(:,merged_ROIs{i}(j)),d1,d2)); 
                            title(sprintf('Component %i',j),'fontsize',16,'fontweight','bold'); axis equal; axis tight;
                    end
                    subplot(1,ln+2,ln+1); imagesc(reshape(Am(:,K_m-length(merged_ROIs)+i),d1,d2));
                            title('Merged Component','fontsize',16,'fontweight','bold');axis equal; axis tight; 
                    subplot(1,ln+2,ln+2);
                        plot(1:T,(diag(max(C(merged_ROIs{i},:),[],2))\C(merged_ROIs{i},:))'); 
                        hold all; plot(1:T,Cm(K_m-length(merged_ROIs)+i,:)/max(Cm(K_m-length(merged_ROIs)+i,:)),'--k')
                        title('Temporal Components','fontsize',16,'fontweight','bold')
                    drawnow;
            end
        end

        %% evaluate components

        options.space_thresh = 0.3;
        options.time_thresh = 0.3;
        [rval_space,rval_time,ind_space,ind_time] = classify_comp_corr(Y,Am,Cm,b,f,options);

        keep = ind_time & ind_space; 
        throw = ~keep;

        if plotoutputs
            figure;
                subplot(121); plot_contours(Am(:,keep),Cn,options,1); title('Selected components','fontweight','bold','fontsize',14);
                subplot(122); plot_contours(Am(:,throw),Cn,options,1);title('Rejected components','fontweight','bold','fontsize',14);
        end

        %% refine estimates excluding rejected components

        Pm.p = p;    % restore AR value
        [A2,b2,C2] = update_spatial_components(Yr,Cm(keep,:),f,[Am(:,keep),b],Pm,options);
        [C2,f2,P2,S2,YrA2] = update_temporal_components(Yr,A2,b2,C2,f,Pm,options);

        % jyr:
        % figure()
        % testim = reshape(P.sn, d1, d2);
        % pixim = reshape(P.pixels, d1, d2);
        % subplot(1,2,1)
        % imshow(testim/max(max(testim))); colorbar()
        % subplot(1,2,2)
        % imshow(pixim/max(max(pixim))); colorbar()
        %% do some plotting

        options.df_prctile = 50;
        options.df_window = 20;

        [A_or,C_or,S_or,P_or] = order_ROIs(A2,C2,S2,P2); % order components
        K_m = size(C_or,1);
        [C_df,~] = extract_DF_F(Yr,A_or,C_or,P_or,options); % extract DF/F values (optional)

        figure;
        [Coor,json_file] = plot_contours(A_or,Cn,options,1); % contour plot of spatial footprints
        %savejson('jmesh',json_file,'filename');        % optional save json file with component coordinates (requires matlab json library)

        close ALL;

    
        nmfstruct.file(fidx).source = source_dir;
        nmfstruct.file(fidx).tiffPath = tif_path;
        nmfstruct.file(fidx).tiffName = tif_name;
        nmfstruct.file(fidx).rois = A_or;
        nmfstruct.file(fidx).correlation = Cn;
        nmfstruct.file(fidx).info = json_file;
        nmfstruct.file(fidx).options = options;
        nmfstruct.file(fidx).preprocessing.scaleFOV = scaleFOV;
        nmfstruct.file(fidx).preprocessing.removeBadFrame = removeBadFrames;
        if removeBadFrames
            nmfstruct.file(fidx).preprocessing.refframe = refframe;
            nmfstruct.file(fidx).preprocessing.badframes = badframes;
            nmfstruct.file(fidx).preprocessing.corrcoeffs = corrs;
        end

        %nmfStructName = sprintf('nmf_Slice%02d_File%03d.mat', currSlice, currFile)
        %save(fullfile(maskPath, nmfStructName), '-struct', 'nmfstruct', '-v7.3');
        %nmfStructNames{end+1} = fullfile(maskPath, nmfStructName);


    end
    
    nmfStructName = sprintf('nmf_Slice%02d.mat', currSlice)
    save(fullfile(maskPath, nmfStructName), '-struct', 'nmfstruct', '-v7.3');
    nmfStructNames{end+1} = fullfile(maskPath, nmfStructName);

    %% display components

end
    %plot_components_GUI(Yr,A_or,C_or,b2,f2,Cn,options)

    % D.maskPaths = nmfStructNames;
    % save(fillfile(D.datastructPath, D.name), '-append', '-struct', 'D');

end