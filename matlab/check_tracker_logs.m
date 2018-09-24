% cd to folder where *.json are placed
data=jsondecode(fileread('davison_tracker_internals.json'))
time=1:data.FramesCount;
%% camera uncertainties
cams=[data.Frames.CamPosUnc_s]';
c0=cams(:,1);
c1=cams(:,2);
c2=cams(:,3);

figure(1)
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

figure(2)
plot(time,sps0,'r')
title('sal pnt median unc')
hold on
plot(time,sps1,'g')
plot(time,sps2,'b')
hold off
%%
xs=[data.Frames.FrameProcessingDur];
xs=[data.Frames.CurReprojErr];
figure(3)
plot(time,xs,'k')
