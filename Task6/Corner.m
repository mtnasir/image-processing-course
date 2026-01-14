% Harris Corner Detection
% Read the image

img = imread('coins.png');
if size(img, 3) == 3
    gray_img = rgb2gray(img);
else
    gray_img = img;
end
gray_img = double(gray_img);

% Parameters
k = 0.04;           % Harris detector free parameter
threshold = 1e6;    % Threshold for corner detection
window_size = 3;    % Window size for computing gradients

% Compute image gradients using Sobel operator
Ix = conv2(gray_img, [-1 0 1; -2 0 2; -1 0 1], 'same');
Iy = conv2(gray_img, [-1 -2 -1; 0 0 0; 1 2 1], 'same');

% Compute products of derivatives
Ix2 = Ix .^ 2;
Iy2 = Iy .^ 2;
Ixy = Ix .* Iy;

% Gaussian filter for smoothing
sigma = 1.5;
filter_size = 2 * ceil(3 * sigma) + 1;
gaussian_filter = fspecial('gaussian', filter_size, sigma);

% Compute components of Harris matrix
Sx2 = conv2(Ix2, gaussian_filter, 'same');
Sy2 = conv2(Iy2, gaussian_filter, 'same');
Sxy = conv2(Ixy, gaussian_filter, 'same');

% Compute Harris response
R = (Sx2 .* Sy2 - Sxy .^ 2) - k * (Sx2 + Sy2) .^ 2;

% Non-maximum suppression
R_max = ordfilt2(R, window_size^2, ones(window_size));
corners = (R == R_max) & (R > threshold);

% Get corner coordinates
[row, col] = find(corners);

% Display results
figure;
imshow(uint8(gray_img));
hold on;
plot(col, row, 'r+', 'MarkerSize', 10, 'LineWidth', 2);
title(['Harris Corner Detection - ', num2str(length(row)), ' corners found']);
hold off;

fprintf('Number of corners detected: %d\n', length(row));