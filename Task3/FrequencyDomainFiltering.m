%% Frequency Domain Filtering using DFT in MATLAB
% - Low-pass filter
% - High-pass filter
% - Notch (band-reject) filter

clear; close all; clc;

%% 1. Read and prepare image
I = imread('cameraman.tif');     % Any grayscale image
I = im2double(I);                % Convert to double [0,1]

[rows, cols] = size(I);

% 2D DFT and shifting zero-frequency component to center
F = fft2(I);
F_shift = fftshift(F);

% Magnitude spectrum (for display)
F_mag = log(1 + abs(F_shift));
F_mag = mat2gray(F_mag);

figure;
subplot(1,2,1); imshow(I,[]); title('Original Image');
subplot(1,2,2); imshow(F_mag,[]); title('Log Magnitude Spectrum');


%% 2. Create frequency grid
% Frequency coordinates (centered at 0 using fftshift convention)
u = -floor(cols/2):floor(cols/2)-1;
v = -floor(rows/2):floor(rows/2)-1;
[U,V] = meshgrid(u, v);

% Distance from the center (radius in frequency domain)
D = sqrt(U.^2 + V.^2);


%% 3. Low-Pass Filter (Ideal or Butterworth)

D0_lp = 40;        % Cutoff radius in frequency "pixels"
n = 2;             % Order for Butterworth (if used)

% (a) Ideal low-pass filter
H_LP_ideal = double(D <= D0_lp);

% (b) Butterworth low-pass filter (more practical, smoother)
H_LP_butter = 1 ./ (1 + (D./D0_lp).^(2*n));

% Apply one of the filters (here we use Butterworth LPF)
G_LP = F_shift .* H_LP_butter;

% Inverse DFT to get filtered image
g_lp = real(ifft2(ifftshift(G_LP)));

figure;
subplot(1,3,1); imshow(H_LP_butter,[]); title('Butterworth LP Filter');
subplot(1,3,2); imshow(g_lp,[]); title('Low-Pass Filtered Image');
subplot(1,3,3); imshow(I - g_lp,[]); title('High-Frequency Residual');


%% 4. High-Pass Filter (from Low-Pass)

D0_hp = 40;    % Same cutoff as low-pass (for comparison)
n_hp  = 2;

% Butterworth high-pass filter: complement of Butterworth low-pass
H_HP_butter = 1 - 1 ./ (1 + (D./D0_hp).^(2*n_hp));

G_HP = F_shift .* H_HP_butter;

g_hp = real(ifft2(ifftshift(G_HP)));

figure;
subplot(1,3,1); imshow(H_HP_butter,[]); title('Butterworth HP Filter');
subplot(1,3,2); imshow(g_hp,[]); title('High-Pass Filtered Image');
subplot(1,3,3); imshow(I + 0.7*g_hp,[]); title('Sharpened (I + k*HP)');


%% 5. Notch (Band-Reject) Filter
% Example: remove periodic noise at certain frequency locations
% Here we create a simple *ideal notch reject* pair at (u0,v0) and (-u0,-v0)

% Choose notch center in frequency coordinates (relative to center)
u0 = 30;     % frequency offset in x (columns) direction
v0 = 0;      % frequency offset in y (rows) direction
D0_notch = 5; % Radius of notch hole

% Distance to two symmetric notch centers
Dk1 = sqrt((U - u0).^2 + (V - v0).^2);
Dk2 = sqrt((U + u0).^2 + (V + v0).^2);

% Ideal notch reject filter (start with all-pass = 1, then zero-out bands)
H_notch = ones(rows, cols);
H_notch(Dk1 <= D0_notch) = 0;
H_notch(Dk2 <= D0_notch) = 0;

% Apply notch filter
G_notch = F_shift .* H_notch;
g_notch = real(ifft2(ifftshift(G_notch)));

figure;
subplot(1,3,1); imshow(H_notch,[]); title('Notch Reject Filter');
subplot(1,3,2); imshow(log(1+abs(G_notch)),[]); title('Spectrum after Notch');
subplot(1,3,3); imshow(g_notch,[]); title('Image after Notch Filtering');