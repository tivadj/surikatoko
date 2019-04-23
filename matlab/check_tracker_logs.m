% cd to folder where *.json are placed
data=jsondecode(fileread('davison_tracker_internals.json'))
time=1:data.FramesCount;
fig=4
%% camera uncertainties
cams=[data.Frames.CamPosUnc_s]';
c0=cams(:,1);
c1=cams(:,2);
c2=cams(:,3);

figure(fig+1)
plot(time,c0,'r')
title('cam unc')
hold on
plot(time,c1,'g')
plot(time,c2,'b')
hold off
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
%%
figure(fig+3)
subplot(5,1,1)
plot(time,[data.Frames.CurReprojErrMeas],'k')
title('ReprErrMeas')

subplot(5,1,2)
plot(time,[data.Frames.EstimatedSalPnts],'k')
title('Estimated salient points count')

subplot(5,1,3)
plot(time,[data.Frames.NewSalPnts],'r')
hold on
plot(time,[data.Frames.CommonSalPnts],'g')
plot(time,[data.Frames.DeletedSalPnts],'b')
hold off
legend('New','Com','Del','Estim')
title('NewComDel salient points count')

subplot(5,1,4)
plot(time,[data.Frames.FrameProcessingDur]*1000,'k')
title('t,ms')

subplot(5,1,5)
plot(time,1./[data.Frames.FrameProcessingDur],'k')
title('fps')

%% check optimal estimate and its error are uncorrelated
figure(fig+4)
subplot(3,1,1)
plot(time,[data.Frames.CurReprojErrMeas],'k')
title('ReprErr Meas')

subplot(3,1,2)
plot(time,[data.Frames.CurReprojErrPred],'c')
title('ReprErr Pred')

subplot(3,1,3)
plot(time,[data.Frames.OptimalEstimMulErr],'k')
title('check estimate and its error are uncorrelated')

