% Load and display original
f = imread('cameraman.tif');
imshow(f) % Fig. 3.25(a).

% Default sharpening
g1 = imsharpen(f);
figure, imshow(g1) % Fig. 3.25(b).

% Increase Gaussian radius
g2 = imsharpen(f, 'Radius', 5);
figure, imshow(g2) % Fig. 3.25(c).

% Increase contrast (Amount)
g3 = imsharpen(f, 'Radius', 5, 'Amount', 1.5);
figure, imshow(g3) % Fig. 3.25(d).

% Stronger contrast
g5 = imsharpen(f, 'Radius', 5, 'Amount', 2);
figure, imshow(g5) % Fig. 3.25(e).

% Add threshold to limit sharpening in flat areas
g6 = imsharpen(f, 'Radius', 5, 'Amount', 2, 'Threshold', 0.5);
figure, imshow(g6) % Fig. 3.25(f).

%%
% Read a sample color image and convert to grayscale (R2019b+: im2gray)
I = imread('peppers.png');           % built-in sample image
fn = im2gray(I);                     % grayscale version (uint8)

% Median filter (3x3 window) with symmetric padding
gm = medfilt2(fn, [3 3], 'symmetric');

% Show before/after
figure;
subplot(1,2,1); imshow(fn); title('Grayscale (fn)');
subplot(1,2,2); imshow(gm); title('Median filtered (gm)');

% Optional: save images
imwrite(fn, 'peppers_gray.png');
imwrite(gm, 'peppers_gray_medfilt3.png');
%%
% Unsharp Masking and Highboost Filtering demo
% - Reads an image (color or grayscale)
% - Creates a blurred version (Gaussian)
% - Forms the mask = original - blur
% - Adds k * mask back (k=1 -> unsharp; k>1 -> highboost)

clear; close all; clc;

% PARAMETERS
imgPath = 'cameraman.tif';  % any image file path; 'cameraman.tif' is built-in and grayscale
sigma   = 1.2;              % Gaussian blur standard deviation (controls amount of smoothing)
k_unsharp  = 1.0;           % Unsharp masking (k = 1)
k_highboost = 1.8;          % Highboost (k > 1). Try 1.5–2.5
useThreshold = false;       % Optional: suppress small-mask noise
T = 0.01;                   % Threshold in [0,1] if useThreshold=true

% 1) Read and convert to grayscale double in [0,1]
I0 = imread(imgPath);
I = im2double( im2gray(I0) );   % handles both RGB and grayscale input

% 2) Blur the original (Gaussian). Filter size ~ 6*sigma
hsize = 2*ceil(3*sigma) + 1;
I_blur = imgaussfilt(I, sigma, 'FilterSize', hsize);

% 3) Form the mask (high-frequency detail)
mask = I - I_blur;

% Optional: suppress tiny values in the mask to avoid boosting noise
if useThreshold
    mask = mask .* (abs(mask) >= T);
end

% >>> Show the mask (signed). [] auto-scales display <<<
figure('Color','w');
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
nexttile; imshow(mask, []);     title('Mask (I - I_{blur}), signed'); colorbar;
nexttile; imshow(abs(mask), []); title('|Mask| (magnitude)');         colorbar;

% 4) Add the mask back (unsharp/highboost)
I_unsharp   = I + k_unsharp   * mask;
I_highboost = I + k_highboost * mask;

% Clip to [0,1]
clip = @(X) min(max(X,0),1);
I_unsharp   = clip(I_unsharp);
I_highboost = clip(I_highboost);

% Display results
figure('Color','w');
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
nexttile; imshow(I);           title('Original (grayscale)');
nexttile; imshow(I_blur);      title(sprintf('Blurred (\\sigma=%.2f)', sigma));
nexttile; imshow(I_unsharp);   title('Unsharp (k=1)');
nexttile; imshow(I_highboost); title(sprintf('Highboost (k=%.1f)', k_highboost));