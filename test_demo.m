clear;
%% load file

addpath(genpath('utilities'));
             
%nam = 'demoMovie.tif';          % insert path to tiff stack here
% tif_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/fov6_retinobar_037Hz_final_bluemask_00002/ch1_slices/';
% nam = strcat(tif_path,'fov6_retinobar_037Hz_final_bluemask_00002.tif #1.tif #13.tif');
% tif_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/fov6_retinobar_037Hz_final_bluemask_00002/';
% nam = strcat(tif_path,'fov6_retinobar_037Hz_final_bluemask_00002_Ch1_Sl13.tif');

% 
% tif_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/NMF/';
% tif_name = 'fov6_retinobar_037Hz_final_bluemask_00002_Channel01_Slice13.tif';
% 
% tif_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/NMF/';
% tif_name = 'fov6_retinobar_037Hz_final_bluemask_00002_Channel01_Slice13.tif';
% 
% tif_path = '/nas/volume1/2photon/RESDATA/20161221_JR030W/retinotopy037Hz/NMF/';
% tif_name = 'fov1_bar037Hz_run4_Slice05_Channel01_File005_scaled.tif';
% tif_name = 'fov1_bar037Hz_run4_Slice05_Channel01_File005_scaled_despeckled_backsub20.tif';
% 
% tif_path = '/nas/volume1/2photon/RESDATA/20161222_JR030W/20161222_JR030W_gratings1/fov1_gratings1_10reps_run1_Slice07_Channel01/';
% tif_name ='fov1_gratings_10reps_run1_Slice07_Channel01_File002_scaled.tif';

% -------------------------------------------------------------------------
% TEFO 
% -------------------------------------------------------------------------
%
tif_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/NMF/';
tif_name = 'fov1_grating2_00001_Channel01_Slice18.tif';
mw_fn_base = '20161219_JR030W_grating2.mat';
%
slice_no = 13;
tif_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/gratings/fov6_gratings_bluemask_5trials_00003/';
tif_name = strcat('ch1_slices/', sprintf('fov6_gratings_bluemask_5trials_00003.tif #1 #%i.tif', slice_no));
mw_fn = strcat(tif_path, 'mw_data/', '20161219_JR030W_gratings_bluemask_5trials_2.mwk');

%
% fov6_rsvp_bluemask_test_10trials_00001:
% Slices:  22 13

slice_no = 13;

tif_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/rsvp/fov6_rsvp_bluemask_test_10trials/fov6_rsvp_bluemask_test_10trials_00001/ch1_slices/';
tif_name = sprintf('fov6_rsvp_bluemask_test_10trials_00001.tif #1.tif #%i.tif', slice_no)
%

% 
ridx = 1;
fidx = 13;
tif_path = '/nas/volume1/2photon/RESDATA/TEFO/20160115_AG33/fov3_gratings1/';
tif_name = strcat('run1_slices/', sprintf('fov3_gratings_0000%i.tif #2.tif #%i.tif', ridx, fidx));

% 
ridx = 1;
fidx = 13;
tif_path = '/nas/volume1/2photon/RESDATA/TEFO/20160118_AG33/fov1_gratings1/';
tif_name = strcat('run1_slices/', sprintf('fov1_gratings_0000%i.tif #2.tif #%i.tif', ridx, fidx));





% -------------------------------------------------------------------------
% NON-TEFO 
% -------------------------------------------------------------------------
%
tif_path = '/nas/volume1/2photon/RESDATA/20161222_JR030W/20161222_JR030W_gratings1/fov1_gratings1_10reps_run1_Slice07_Channel01/NMF/';
tif_name = 'fov1_gratings_10reps_run1_Slice07_Channel01_File002_scaled.tif';

%
tif_path = '/nas/volume1/2photon/RESDATA/20161221_JR030W/rsvp/fov2_rsvp_25reps_Slice06_Channel01_File004/';
tif_name = 'fov2_rsvp_25reps_Slice06_Channel01_File004_scaled.tif'


% CONTROLS:
%
slice_no = 13;
tif_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/rsvp/fov6_rsvp_bluemask_shutteroff/ch1_slices/';
tif_name = sprintf('fov6_rsvp_bluemask_shutteroff_00001.tif #1.tif #%i.tif', slice_no)


%%

nam = strcat(tif_path, tif_name);

sframe=1;						% user input: first frame to read (optional, default 1)
%num2read=2000;					% user input: how many frames to read   (optional, default until the end)
Y = bigread2(nam,sframe);

% load(strcat(tif_path, 'Y_highpas.mat'));
% load(strcat(tif_path, 'Y_rollavg_5.mat'));
% Y = reshape(frameArray, 512, 1024, 256);

%Y = Y - min(Y(:)); 
if ~isa(Y,'double');    Y = double(Y);  end         % convert to single

[d1,d2,T] = size(Y);                                % dimensions of dataset
d = d1*d2;                                          % total number of pixels

%% Set parameters

K = 150; %50; %300; %50; %150; %35;                                      % number of components to be found
tau = .5; %2; %4;                                      % std of gaussian kernel (size of neuron) 

% tau = [1 1; 2 2];
% K = [100; 50];

p = 2;     % order of autoregressive system (p = 0 no dynamics, p=1 just decay, p = 2, both rise and decay)
merge_thr = 0.8;                                  % merging threshold

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
%% Data pre-processing

% High-pass filter:
winsize = 25;
for i=119:d1
   for j=396:d2
       tcurr = Y(i,j,:);
       s1 = smooth(squeeze(tcurr), winsize, 'rlowess');
       t1 = squeeze(tcurr) - s1;
       Y(i,j,:) = t1;
   end
end

frameArray = reshape(Y, d, size(Y,3));
framesToAvg = 5;
for pix = 1:length(frameArray)
    tmp0=frameArray(pix,:);
    tmp1=padarray(tmp0,[0 framesToAvg],tmp0(1),'pre');
    tmp1=padarray(tmp1,[0 framesToAvg],tmp0(end),'post');
    rollingAvg=conv(tmp1,fspecial('average',[1 framesToAvg]),'same');%average
    rollingAvg=rollingAvg(framesToAvg+1:end-framesToAvg);
    frameArray(pix,:) = tmp0 - rollingAvg;
end

save(strcat(tif_path, sprintf('Y_rollavg_%i.mat', framesToAvg)), 'frameArray');

figure(); plot(tmp0,'k'); hold on;
plot(rollingAvg, 'r');
hold on;
plot(tmp0-rollingAvg, 'g');


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

save(strcat(tif_path,'Y_highpas.mat'), 'Y');
    
%% fast initialization of spatial components using greedyROI and HALS

[P,Y] = preprocess_data(Y,p);


[Ain,Cin,bin,fin,center] = initialize_components(Y,K,tau,options,P);  % initialize

% display centers of found components
Cn =  correlation_image(Y); %reshape(P.sn,d1,d2);  %max(Y,[],3); %std(Y,[],3); % image statistic (only for display purposes)
figure;imagesc(Cn);
    axis equal; axis tight; hold all;
    scatter(center(:,2),center(:,1),'mo');
    title('Center of ROIs found from initialization algorithm');
    drawnow;

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
display_merging = 1; % flag for displaying merging example
if and(display_merging, ~isempty(merged_ROIs))
    i = 1; %randi(length(merged_ROIs));
    ln = length(merged_ROIs{i});
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

%% evaluate components

options.space_thresh = 0.3;
options.time_thresh = 0.3;
[rval_space,rval_time,ind_space,ind_time] = classify_comp_corr(Y,Am,Cm,b,f,options);

keep = ind_time & ind_space; 
throw = ~keep;
figure;
    subplot(121); plot_contours(Am(:,keep),Cn,options,1); title('Selected components','fontweight','bold','fontsize',14);
    subplot(122); plot_contours(Am(:,throw),Cn,options,1);title('Rejected components','fontweight','bold','fontsize',14);

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

[A_or,C_or,S_or,P_or] = order_ROIs(A2,C2,S2,P2); % order components
K_m = size(C_or,1);
[C_df,~] = extract_DF_F(Yr,A_or,C_or,P_or,options); % extract DF/F values (optional)

figure;
[Coor,json_file] = plot_contours(A_or,Cn,options,1); % contour plot of spatial footprints
%savejson('jmesh',json_file,'filename');        % optional save json file with component coordinates (requires matlab json library)

%% display components

plot_components_GUI(Yr,A_or,C_or,b2,f2,Cn,options)

% -----------------------------------------------------------
% jyr
% -----------------------------------------------------------

% Get MW and ARD info from python .mat file:
pymat_fn = '20161219_JR030W_grating2'; %'fov1_gratings_10reps_run1.mat';
mw_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/NMF/';

pymat_fn = '20161219_JR030W_gratings_bluemask_5trials_2.mat';
mw_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/gratings/fov6_gratings_bluemask_5trials_00003/mw_data/';
%

pymat_fn = '20161219_JR030W_rsvp_bluemask_test_10trials.mat';
mw_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/rsvp/fov6_rsvp_bluemask_test_10trials/fov6_rsvp_bluemask_test_10trials_00001/';
%
pymat_fn = 'fov1_gratings_10reps_run1.mat';
mw_path = '/nas/volume1/2photon/RESDATA/20161222_JR030W/20161222_JR030W_gratings1/fov1_gratings1_10reps_run1_Slice07_Channel01/NMF/';
%
pymat_fn = '20161219_JR030W_grating2.mat';
mw_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/NMF/';
mw_fidx = 1;
%
pymat_fn = '20161219_JR030W_rsvp_bluemask_test_10trials.mat';
mw_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/rsvp/fov6_rsvp_bluemask_test_10trials/fov6_rsvp_bluemask_test_10trials_00001/';
mw_fidx = 1;
%

pymat_fn = '20161221_JR030W_rsvp_25reps.mat';
mw_path = '/nas/volume1/2photon/RESDATA/20161221_JR030W/rsvp/';
% mw_fidx = 4;

%% Load pymat info:

S = load(strcat(mw_path, pymat_fn));

ard_filedurs = double(S.ard_file_durs);
mw_times = S.mw_times_by_file;
offsets = double(S.offsets_by_file);
mw_codes = S.mw_codes_by_file;
mw_file_durs = double(S.mw_file_durs);

ntrash = 0; %4;
tif_no = 1; %8;
mc_file_no = 1; %4;
mw_fidx = mc_file_no+ntrash

if strcmp(class(mw_times), 'int64')
    % GET MW rel times from match_triggers.py: 'rel_times'
    curr_mw_times = double(mw_times(mw_fidx, :)); 
    
    % Get MW codes corresponding to time points:
    curr_mw_codes = mw_codes(mw_fidx, :); 
else
    curr_mw_times = double(mw_times{mw_fidx}); % convert to sec
    curr_mw_codes = mw_codes{mw_fidx};
end
mw_rel_times = ((curr_mw_times - curr_mw_times(1)) + offsets(mw_fidx)); % into seconds
mw_sec = mw_rel_times/1000000;


% Get correct time dur to convert frames to time (s):
curr_mw_filedur = mw_file_durs(mw_fidx)/1000000; % convert to sec

nframes = size(Y, 3);

if ~exists(ard_filedurs)
    y_sec = [0:nframes-1].*(curr_mw_filedur/nframes);
else
%     if strcmp(file_no, 'file13')
%         y_sec1 = [0:nframes-1].*(ard_filedurs(mw_fidx)/nframes);
%         y_sec1 = y_sec1(1:2:end);
%         y_int = diff(y_sec1);
%         y_int = y_int(1);
%         y_sec2 = y_sec1 + y_sec1(end);
%         y_sec = [y_sec1 y_sec2(2:end) y_sec2(end)+y_int];
%     else
        y_sec = [0:nframes-1].*(ard_filedurs(mw_fidx)/nframes); % y_sec = [0:nframes-1]./acquisition_rate;
        % ^^ this doesn't work for tifs where mw ends before end of file
%     end
end

% Get n-stim color-code: -- get 'mw_codes' from match_triggers.py:
nstimuli = 35; %length(unique(mw_codes(1,:)))-1;

nstimuli = 12;


colors = zeros(nstimuli,3);
for c=1:nstimuli
    colors(c,:,:) = rand(1,3);
end

%plot(mw_rel_times, zeros(size(mw_rel_times)), 'r*'); 
y1=get(gca,'ylim');

%%
[guiT,guiY_r, guiC, guiDf] = plot_components_GUI(Yr,A_or,C_or,b2,f2,Cn,options);

%plot_components_GUI(Yr,A_or,C_or,b2,f2,Cn,options)

% plot_components_GUI(Y,A,C,b,f,Cn,options)

% T = size(C_or,2);
% if ndims(Yr) == 3
%     Yr = reshape(Yr,d1*d2,T);
% end
% % if nargin < 6 || isempty(Cn);
% %     Cn = reshape(mean(Y,2),d1,d2);
% % end
% b2 = double(b2);
% C_or = double(C_or);
% f2 = double(f2);
% nA = full(sqrt(sum(A_or.^2))');
% [K,~] = size(C_or);
% A_or = A_or/spdiags(nA,0,K,K);    % normalize spatial components to unit energy
% C_or = bsxfun(@times,C_or,nA(:)); %spdiags(nA,0,K,K)*C;
% 
% 
% AY = A_or'*Yr;
% 
% Y_r = (AY- (A_or'*A_or)*C_or - full(A_or'*double(b2))*f2) + C_or;
% [~,Df] = extract_DF_F(Yr,A_or,C_or,[],options);
% step = 5e3;
% 
% % 
% figure()
% 
% nr = size(A_or,2);     % number of ROIs
% nb = size(f2,1);     % number of background components
% 
% ROIs = [11, 13, 45, 47, 54, 56, 61:66, 69, 70, 73];
% 


% y_frames = y_sec./(curr_mw_filedur/nframes);
% mw_frames = mw_sec./((curr_mw_filedur)/nframes);


for r=1:5 %length(ROIs)
    
    i=ROIs(r);
    
    %%
    % FILE: rsvp? %102; %74 %73; %71; %67; %64; %54; %23; %11
    % FILE: gratings: fov6_gratings_bluemask_5trials_00003 -- 1%10 %5 %4 %3 %1
    % FILE:  gratings2 --2016/12/19_jr030W_/NMF/ -- high-pass.
    i =  13 %54 %24 %11
    
%     plot(1:T,Y_r(i,:)/Df(i),'linewidth',2); hold all; plot(1:T,C(i,:)/Df(i),'linewidth',2);
%     plot(1:guiT,guiY_r(i,:)/guiDf(i), 'k', 'linewidth',.5); hold all; 
%     plot(1:guiT,guiC(i,:)/guiDf(i), 'k', 'linewidth',2);

    plot_all = 1;
    
    if plot_all==1
        figure();
        %plot(y_sec,guiY_r(i,:)/guiDf(i), 'k', 'linewidth',.2); hold all; 
        plot(y_sec,guiC(i,:)/guiDf(i), 'k', 'linewidth',2);

        y1=get(gca,'ylim');
        %y1 = [-0.1 0.2]; %get(gca,'ylim');
        title(sprintf('Component %i (calcium DF/F value)',i),'fontsize',16,'fontweight','bold');
        %leg = legend('Raw trace (filtered)','Inferred');
        %set(leg,'FontSize',14,'FontWeight','bold');
        xlim([0 400]) %550]);
        drawnow;

        legend_labels = {};
        l=1;
        hold on;
        for stim=1:2:(length(mw_rel_times)-1)
           sx = [mw_sec(stim) mw_sec(stim+1) mw_sec(stim+1) mw_sec(stim)];
           %sx = [mw_frames(stim) mw_frames(stim+1) mw_frames(stim+1) mw_frames(stim)];
           y1=get(gca,'ylim');
           sy = [y1(1) y1(1) y1(2) y1(2)];
           %sy = [-0.1 -0.1 0.1 0.1].*100;
           curr_code = curr_mw_codes(stim);
           patch(sx, sy, colors(curr_code,:,:), 'FaceAlpha', 0.5, 'EdgeAlpha', 0)
           hold on;
           %text(sx(1), sy(1)+abs(sy(1))*0.5, num2str(curr_code));
           text(sx(1), sy(3)-abs(sy(2))*0.5, num2str(curr_code));
           hold on;
           if stim==1
               legend_labels{1} = {num2str(curr_code)};
           else
            legend_labels{1} = [legend_labels{1} {num2str(curr_code)}];
           end
        end
    else
        
        
        ntrials_per_chunk = 40;
        mw_chunk = mw_sec;
        while mod(length(mw_chunk), ntrials_per_chunk) > 0
            mw_chunk = padarray(mw_chunk, [0 1], mw_sec(end), 'post');
        end
        ca_chunk = y_sec;
        while mod(length(y_sec), ntrials_per_chunk) > 0
            ca_chunk = padarray(ca_chunk, [0 1], mw_sec(end), 'post');
        end     
            
        mw_chunks = mat2cell(mw_chunk,1,repmat(ntrials_per_chunk*2, [1 length(mw_chunk)/(ntrials_per_chunk*2)]));
        
        for chunk=1:length(mw_chunks)
            %%
           figure(); %Chunks3-4
           chunk = 1
           
           tmp = abs(y_sec-mw_chunks{chunk}(1)); % Find closest matching Ca trace idx for MW trial
           [idx idx] = min(tmp); %index of closest value
           closest = y_sec(idx); %closest value
           start_chunk = idx;
           tmp = abs(y_sec-mw_chunks{chunk+1}(1)); % Find closest matching of end of last MW trial in chunk
           [idx idx] = min(tmp); %index of closest value
           closest = y_sec(idx); %closest value
           end_chunk = idx;
           
           plot(y_sec(start_chunk:end_chunk),guiC(i,start_chunk:end_chunk)/guiDf(i), 'k', 'linewidth',2);
           title(sprintf('Component %i (calcium DF/F value)',i),'fontsize',16,'fontweight','bold');
           %leg = legend('Raw trace (filtered)','Inferred');
           %set(leg,'FontSize',14,'FontWeight','bold');
           drawnow;
           hold on;
           y1=get(gca,'ylim');
           
           mwidx = (chunk-1)*(ntrials_per_chunk*2)+1;
           for stim=mwidx:2:(mwidx+ntrials_per_chunk*2)-1
               sx = [mw_sec(stim) mw_sec(stim+1) mw_sec(stim+1) mw_sec(stim)];
               sy = [y1(1) y1(1) y1(2) y1(2)];
               curr_code = curr_mw_codes(stim);
               patch(sx, sy, colors(curr_code,:,:), 'FaceAlpha', 0.5, 'EdgeAlpha', 0)
               hold on;
               text(sx(1), sy(1)+abs(sy(1))*0.5, num2str(curr_code));
               hold on;
               if stim==1
                   legend_labels{1} = {num2str(curr_code)};
               else
                legend_labels{1} = [legend_labels{1} {num2str(curr_code)}];
               end
           end
           %%
        end
        
    end
    
    %%

%     h=legend(legend_labels{1}, 'location', 'northeast')
%     set(h, 'position', [0.95 0.3 0.005 0.01])

    
end

%



y_frames = y_sec./(curr_mw_filedur/nframes);
mw_frames = mw_sec./(curr_mw_filedur/nframes);

legend_labels = {};
l=1;
hold on;
for stim=1:2:length(mw_rel_times)-1
   %sx = [mw_sec(stim) mw_sec(stim+1) mw_sec(stim+1) mw_sec(stim)];
   sx = [mw_frames(stim) mw_frames(stim+1) mw_frames(stim+1) mw_frames(stim)];
   sy = [y1(1) y1(1) y1(2) y1(2)];
   %sy = [-0.1 -0.1 0.1 0.1].*100;
   curr_code = curr_mw_codes(stim);
   patch(sx, sy, colors(curr_code,:,:), 'FaceAlpha', 0.5, 'EdgeAlpha', 0)
   hold on;
   if stim==1
       legend_labels{1} = {num2str(curr_code)};
   else
    legend_labels{1} = [legend_labels{1} {num2str(curr_code)}];
   end
end 

%% make movie

make_patch_video(A_or,C_or,b2,f2,Yr,Coor,options)