vid = videoinput('winvideo', 1, 'YUY2_640x480');
vid.TriggerRepeat = 100;
vid.FrameGrabInterval = 5;
start(vid);
while (vid.FramesAvailable >= 2)
   frames = getdata(vid, 2);
   diff_im = imabsdiff(frames(:,:,:,1), frames(:,:,:,2));
   imshow(diff_im);
   drawnow;
end
stop(vid);
delete(vid);
clear vid;