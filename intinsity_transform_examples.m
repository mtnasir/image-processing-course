% Reset workspace, command window, and close figures
clear
clc
close all

% Read grayscale demo image and display it
I = imread('pout.tif');
imshow(I)

%% Gamma/intensity transformation
% g = imadjust(f, [lowIn highIn], [lowOut highOut], gamma);
% Below: map input intensities in [0.3, 0.7] to full display range; outside values are clipped
K = imadjust(I, [0.3 0.7], []);
figure
imshow(K)

%%% Use stretchlim to estimate input limits automatically (≈1% saturation on each end by default)
J = imadjust(I, stretchlim(I), []);
figure
imshow(J)

%% Image complement (invert intensities)
clear
clc
close all
bw = imread('text.png');          % Binary or grayscale text image
bw2 = imcomplement(bw);           % Invert: 0↔1 for logical, 0↔255 for uint8
imshowpair(bw, bw2, 'montage')    % Side-by-side comparison

%% Histogram equalization and visualization
I = imread('tire.tif');           % Low-contrast image
J = histeq(I);                    % Enhance contrast by flattening histogram
imshowpair(I, J, 'montage')
axis off
figure
imhist(I, 64)                     % Histogram of original (64 bins)
figure
imhist(J, 64)                     % Histogram after equalization

[J, T] = histeq(I);               % Also return the mapping T (CDF-based transform)
figure
plot((0:255)/255, T);             % Plot input→output intensity mapping in [0,1]