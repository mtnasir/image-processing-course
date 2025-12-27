%% Simple Segmentation Examples in MATLAB
% Requires: Image Processing Toolbox

clear; clc; close all;

%% Load image (RGB) and convert to grayscale
I = imread('peppers.png');       % Replace with: imread('yourImage.png')
I = im2uint8(I);                 % Ensure 8-bit
Igray = rgb2gray(I);

figure; imshow(I); title('Original RGB Image');


%% 1) REGION GROWING (using grayconnected)
% Use center pixel as seed and grow region based on intensity tolerance.

seedRow = round(size(Igray,1)/2);
seedCol = round(size(Igray,2)/2);
tolerance = 15;   % try 5–30

regionMask = grayconnected(Igray, seedRow, seedCol, tolerance);

figure;
imshow(regionMask);
title('Region Growing - Binary Mask');

% Overlay mask on original image
RG_overlay = I;
RG_overlay(repmat(~regionMask,[1 1 3])) = 0;

figure;
imshow(RG_overlay);
title('Region Growing - Segmented Region');


%% 2) K-MEANS CLUSTERING (on RGB pixels)
% Cluster pixels in RGB space into k clusters.

k = 3;  % number of clusters

imgReshaped = double(reshape(I, [], 3));  % N x 3 matrix of RGB pixels

[idx, C] = kmeans(imgReshaped, k, ...
                  'Distance','sqeuclidean', ...
                  'Replicates',3);

pixelLabels = reshape(idx, size(I,1), size(I,2));

figure;
imshow(label2rgb(pixelLabels));
title('K-means Clustering (k = 3)');


%% 3) SUPERPIXEL SEGMENTATION
% Divide image into superpixels and show their boundaries.

numSuperpixels = 50;

[L, numLabels] = superpixels(I, numSuperpixels);

BW = boundarymask(L);

figure;
imshow(imoverlay(I, BW, 'cyan'));
title(['Superpixels (N = ' num2str(numLabels) ')']);


%% 4) GABOR FILTER–BASED SEGMENTATION
% Apply a bank of Gabor filters and use the sum of magnitudes as a texture map,
% then threshold it.

wavelengths = [4 8];                   % spatial wavelengths
orientations = 0:45:135;               % orientations in degrees
g = gabor(wavelengths, orientations);  % Gabor filter bank

gaborMag = imgaborfilt(Igray, g);      % H x W x numFilters

% Combine filter responses (sum over filters)
gaborFeat = sum(gaborMag, 3);
gaborFeat = mat2gray(gaborFeat);       % normalize to [0,1]

% Simple threshold using Otsu
level = graythresh(gaborFeat);
gaborMask = imbinarize(gaborFeat, level);

figure;
subplot(1,2,1); imshow(gaborFeat,[]); title('Gabor Feature Map');
subplot(1,2,2); imshow(gaborMask);    title('Gabor-based Segmentation Mask');


%% 5) WATERSHED SEGMENTATION
% Apply watershed on gradient magnitude to segment regions.

% Compute gradient magnitude
gmag = imgradient(Igray);

figure;
imshow(gmag, []);
title('Gradient Magnitude');

% Watershed transform
Lw = watershed(gmag);

LwRGB = label2rgb(Lw, 'jet', 'w', 'shuffle');

figure;
imshow(LwRGB);
title('Watershed Segmentation (Labels)');

% Optional: show boundaries on original
wBoundaries = Lw == 0;
I_watershed = I;
I_watershed(repmat(wBoundaries,[1 1 3])) = 255;

figure;
imshow(I_watershed);
title('Watershed Boundaries on Original Image');