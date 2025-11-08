% image processng spatial filters:
% Read a grayscale image (built-in example)
A = imread('cameraman.tif');     % already grayscale
A = im2double(A);                % work in double to avoid clipping

% 1) Low-pass (smoothing) — 3x3 average filter
h_lp = (1/9) * ones(3);
B_lp = imfilter(A, h_lp, 'replicate', 'conv');  % B = imfilter(A, h)

% 2) High-pass — difference of impulse and low-pass (zero DC)
delta = [0 0 0; 0 1 0; 0 0 0];
h_hp = delta - h_lp;                              % high-pass kernel
B_hp = imfilter(A, h_hp, 'replicate', 'conv');    % may have negative values
B_hp_disp = mat2gray(B_hp);                       % for display

% 3) Laplacian — 4-neighbor discrete Laplacian
h_lap = [ 0 -1  0;
         -1  4 -1;
          0 -1  0 ];
B_lap = imfilter(A, h_lap, 'replicate', 'conv');  % Laplacian response (signed)
B_lap_disp = mat2gray(B_lap);                     % for display

% 4) Edge detection — Sobel (gradient magnitude)
h_sobel_x = [-1 0 1; -2 0 2; -1 0 1];
h_sobel_y = h_sobel_x.';
Gx = imfilter(A, h_sobel_x, 'replicate', 'conv');
Gy = imfilter(A, h_sobel_y, 'replicate', 'conv');
B_edge = mat2gray(sqrt(Gx.^2 + Gy.^2));           % gradient magnitude

% 5) Sharpening — unsharp masking implemented as a single kernel
%    h_sharp = (1+alpha)*delta - alpha*h_lp
alpha = 1.0;                                      % amount of sharpening
h_sharp = (1+alpha)*delta - alpha*h_lp;
B_sharp = imfilter(A, h_sharp, 'replicate', 'conv');

% (Alternative sharpening via Laplacian in one pass)
%    g = f - k * (∇^2 f)  =>  h = delta - k*h_lap
k = 0.2;
h_sharp_lap = delta - k*h_lap;
B_sharp2 = imfilter(A, h_sharp_lap, 'replicate', 'conv');


% Visualize
figure;
subplot(2,3,1), imshow(A),            title('Original');
subplot(2,3,2), imshow(B_lp),         title('Low-pass (3x3 average)');
subplot(2,3,3), imshow(B_hp_disp),    title('High-pass (delta - LP)');
subplot(2,3,4), imshow(B_lap_disp),   title('Laplacian response');
subplot(2,3,5), imshow(B_edge),       title('Edges (Sobel magnitude)');
subplot(2,3,6), imshow(B_sharp),      title('Sharpened (Unsharp, \alpha=1)');


% 6) Median filtering — great for salt & pepper noise
A_sp = imnoise(A, 'salt & pepper', 0.05);     % add 5% impulse noise

% Apply median filters with different neighborhood sizes
B_med3 = medfilt2(A_sp, [2 2], 'symmetric');  % small window
B_med5 = medfilt2(A_sp, [5 5], 'symmetric');  % stronger denoising

% Compare
figure;
subplot(1,4,1), imshow(A),      title('Original');
subplot(1,4,2), imshow(A_sp),   title('Noisy (S&P 5%)');
subplot(1,4,3), imshow(B_med3), title('Median 2x2 (symmetric)');
subplot(1,4,4), imshow(B_med5), title('Median 5x5 (symmetric)');