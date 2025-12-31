clear
close all
img=imread("cameraman.tif");

imgg=im2gray(img);

imedg1=edge(img,"sobel");

imshowpair(img,imedg1,'montage')
title("Sobel edge detection")

figure
imedg2=edge(img,'canny'.'',0.2);
imshowpair(img,imedg2,'montage')
title("Canny edge detection")

%% Thresholding
figure
img= imread('coins.png');

imgg=im2gray(img);
imhist(imgg)
figure
imgTh=imgg>90;
imshowpair(img,imgTh,'montage')
title("Manual thersoholding")
figure

T=graythresh(imgg);
imgTh=imgg>T*255;
T*255
imshowpair(img,imgTh,'montage')
figure
BW = imbinarize(img,T);
imshowpair(img,BW,'montage')
title("Otus's thersoholding")

figure
BW = imbinarize(img,'adaptive');
imshowpair(img,BW,'montage')
title("Adaptive thersoholding")

%% K means
pixelsA=double(reshape(imgg, [], 1));

[ind, KmeansC]=kmeans(pixelsA,2);

imgKmeans= reshape(uint8(KmeansC(ind)), size(imgg));
imshowpair(img,imgKmeans,'montage')
title("Kmeans Segmentation")


