%% Example 2: Add sinusoidal noise and remove it with a notch filter

clear; close all; clc;

%% 1. Read base image
I = imread('cameraman.tif');   % Any grayscale image
I = im2double(I);
[rows, cols] = size(I);

figure;
imshow(I,[]);
title('Original Image');

%% 2. Create synthetic sinusoidal (periodic) noise
% Frequency parameters (in cycles across the image)
u0 = 15;   % horizontal frequency component (across columns)
v0 = 0;    % vertical frequency component  (across rows)
A  = 0.3;  % amplitude of the sinusoidal noise

% Build coordinate grids (0-based indices to match DFT bins nicely)
x = 0:cols-1;
y = 0:rows-1;
[X, Y] = meshgrid(x, y);

% 2D sinusoidal pattern: sin(2*pi*(u0*x/cols + v0*y/rows))
noise = A * sin(2*pi*(u0*X/cols + v0*Y/rows));

% Add noise to the image
I_noisy = I + noise;
I_noisy = mat2gray(I_noisy);   % ensure in [0,1]

figure;
subplot(1,2,1); imshow(noise,[]);    title('Added Sinusoidal Noise');
subplot(1,2,2); imshow(I_noisy,[]);  title('Noisy Image');

%% 3. DFT of the noisy image
F_noisy       = fft2(I_noisy);
F_noisy_shift = fftshift(F_noisy);

F_noisy_mag = log(1 + abs(F_noisy_shift));
F_noisy_mag = mat2gray(F_noisy_mag);

figure;
imshow(F_noisy_mag,[]);
title('Log Magnitude Spectrum of Noisy Image');


%% 4. Design a notch reject filter at the known noise frequencies
% Build frequency coordinate grid (centered)
u = -floor(cols/2):floor(cols/2)-1;
v = -floor(rows/2):floor(rows/2)-1;
[U, V] = meshgrid(u, v);

% Distance from the two symmetric noise peaks in the shifted spectrum
% Peaks are located around (u0, v0) and (-u0, -v0)
D0      = 4;   % radius of each notch (in frequency pixels)
Dk1     = sqrt((U - u0).^2 + (V - v0).^2);
Dk2     = sqrt((U + u0).^2 + (V + v0).^2);

% Ideal notch reject filter (1 everywhere except small zeros at noise peaks)
H_notch = ones(rows, cols);
H_notch(Dk1 <= D0) = 0;
H_notch(Dk2 <= D0) = 0;

figure;
imshow(H_notch,[]);
title('Notch Reject Filter (Ideal)');


%% 5. Apply the notch filter in the frequency domain
G_clean_shift = F_noisy_shift .* H_notch;
G_clean       = ifftshift(G_clean_shift);
g_clean       = real(ifft2(G_clean));    % Back to spatial domain

% For comparison, also look at spectrum after filtering
G_clean_mag = log(1 + abs(G_clean_shift));
G_clean_mag = mat2gray(G_clean_mag);

figure;
subplot(2,2,1); imshow(I_noisy,[]);     title('Noisy Image');
subplot(2,2,2); imshow(F_noisy_mag,[]); title('Noisy Spectrum');
subplot(2,2,3); imshow(G_clean_mag,[]); title('Filtered Spectrum');
subplot(2,2,4); imshow(g_clean,[]);     title('After Notch Filtering');

