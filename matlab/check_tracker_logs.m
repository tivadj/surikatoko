% cd to folder where *.json are placed
data=jsondecode(fileread('davison_tracker_internals.json'))
time=1:data.FramesCount;
fig=4;
fprintf(1, "%s\n", datetime())
%% check optimal estimate and its error are uncorrelated
figure(fig+2)

subplot(3,1,1)
err_pred = [data.Frames.CurReprojErrPred]./[data.Frames.EstimatedSalPnts];
err_pred([data.Frames.CurReprojErrPred]==-1) = NaN;
plot(time, err_pred,'c')
title('ReprErr Pred, pix')

subplot(3,1,2)
err_meas = [data.Frames.CurReprojErrMeas]./[data.Frames.EstimatedSalPnts];
err_meas([data.Frames.CurReprojErrPred]==-1) = NaN;
plot(time,err_meas,'k')
title('ReprErr Meas, pix')

subplot(3,1,3)
plot(time,[data.Frames.OptimalEstimMulErr],'k')
title('check estimate and its error are uncorrelated')
%% camera position and it's error
cam=[data.Frames.CamState]';
cam_gt=[data.Frames.CamStateGT]';

figure(fig+3)
subplot(2,1,1)
plot(time,cam(:,1),'r')
title('cam pos, m')
hold on
plot(time,cam(:,2),'g')
plot(time,cam(:,3),'b')
plot(time,cam_gt(:,1),'r--')
plot(time,cam_gt(:,2),'g:')
plot(time,cam_gt(:,3),'b-.')
hold off

cam_err=[data.Frames.EstimErr]';
cam_err_std=[data.Frames.EstimErrStd]';
subplot(2,1,2)
plot(time,cam_err(:,1),'r')
title('cam error')
hold on
plot(time,cam_err(:,2),'g')
plot(time,cam_err(:,3),'b')
plot(time,cam_err_std(:,1),'r-.',time,-cam_err_std(:,1),'r-.')
plot(time,cam_err_std(:,2),'g-.',time,-cam_err_std(:,2),'g-.')
plot(time,cam_err_std(:,3),'b-.',time,-cam_err_std(:,3),'b-.')
hold off
%% state and ground truth state
cam=[data.Frames.CamState]';
cam_gt=[data.Frames.CamStateGT]';
cam_err=[data.Frames.EstimErr]';
cam_err_std=[data.Frames.EstimErrStd]';
figure(fig+4)
clf;
if ~exist('estimstatetoshow')
    estimstatetoshow=1:size(cam,2); % all
    estimstatetoshow=[1,2,3,8,9,10];
end
subvw=[length(estimstatetoshow),1];
usesubview=1;
for i = estimstatetoshow
    subplot(subvw(1),subvw(2),usesubview);
    usesubview = usesubview + 1;
    plot(time,cam(:,i),'k');    
    hold on;
    plot(time,cam_gt(:,i),'m');
    hold off;
    title(sprintf('state #%d, m',i));
end
%% check error in estimate and it's std
estim_errs=[data.Frames.EstimErr]';
estim_errs_sig=[data.Frames.EstimErrStd]';
n=size(estim_errs,2);
figure(fig+5)
clf
if ~exist('estimerrtoshow')
    estimerrtoshow=1:13;
    estimerrtoshow=1:n; % all
end
subvw=[length(estimerrtoshow),1];
usesubview=1;
for statei = estimerrtoshow
    ei=estim_errs(:,statei);
    esig=estim_errs_sig(:,statei);
    subplot(subvw(1),subvw(2),usesubview);
    usesubview = usesubview + 1;
    plot(1:size(ei),ei, 'k')
    title(sprintf('err in estim #%d, m or rad',statei))
    hold on
    plot(1:size(ei),esig, 'c')
    plot(1:size(ei),-esig, 'c')
    hold off
end
%% check residual and its theoretical bounds
resids=[data.Frames.MeasResidual]';
resids_sig=[data.Frames.MeasResidualStd]';
n=size(resids,2);
figure(fig+6)
clf
if ~exist('residtoshow')
    residtoshow=1:n; % all
end
subvw=[length(residtoshow),1];
usesubview=1;
for statei = residtoshow
    ei=resids(:,statei);
    ei(1)=0;
    esig=resids_sig(:,statei);
    esig(1)=0;
    subplot(subvw(1),subvw(2),usesubview);
    usesubview = usesubview + 1;
    plot(1:size(ei),ei, 'k')
    hold on
    plot(1:size(ei),esig, 'c')
    plot(1:size(ei),-esig, 'c')
    hold off
    title(sprintf('%d',statei))
end
%% dynamics of new-common-deleted salient points
figure(fig)
clf;
subplot(4,1,1)
plot(time,[data.Frames.CurReprojErrMeas]./[data.Frames.EstimatedSalPnts],'k')
title('ReprErrMeas')

subplot(4,1,2)
plot(time,[data.Frames.NewSalPnts],'r')
hold on
plot(time,[data.Frames.CommonSalPnts],'g')
plot(time,[data.Frames.DeletedSalPnts],'b')
plot(time,[data.Frames.EstimatedSalPnts],'k')
hold off
legend('New','Com','Del','Estim')
title('NewComDelEst salient points count')

subplot(4,1,3)
plot(time,[data.Frames.FrameProcessingDur]*1000,'k')
title('t,ms')

subplot(4,1,4)
plot(time,1./[data.Frames.FrameProcessingDur],'k')
title('fps')

%% salient points uncertainties
sps=[data.Frames.SalPntUncMedian_s]';
sps0=sps(:,1);
sps1=sps(:,5);
sps2=sps(:,9);

figure(fig+2)
plot(time,sps0,'r')
title('sal pnt median unc')
hold on
plot(time,sps1,'g')
plot(time,sps2,'b')
hold off
