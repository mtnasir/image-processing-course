%% example1
%% Combining spatial enhancement methods (as in the figures)
% Replace with your image filename
imgPath = 'bone_scan.png';

% Read and convert to [0,1] grayscale
I = imread(imgPath);
if size(I,3) == 3
    I = rgb2gray(I);
end
I = im2double(I);

%% (b) Laplacian of (a) using the 8-neighbor kernel (Fig. 3.51(d))
%   [ 1  1  1
%     1 -8  1
%     1  1  1 ]
hLap = [1 1 1; 1 -8 1; 1 1 1];
L = imfilter(I, hLap, 'replicate', 'conv');     % Laplacian response
L_disp = mat2gray(L);                           % scaled for display only

%% (c) Sharpened image obtained by adding (a) and (b)
% The text/caption say to add the Laplacian to the original.
I_sharp = I + L;
I_sharp = mat2gray(I_sharp);                    % clip/scale to [0,1]

%% (d) Sobel gradient magnitude of (a)
[Gx, Gy] = imgradientxy(I, 'sobel');
[Gmag, ~] = imgradient(Gx, Gy);
Gmag_disp = mat2gray(Gmag);                     % scaled for display

%% Combined enhancement (as described in the text)
% Smooth the gradient to suppress noise, then use it as a mask
% (masking = pointwise multiplication).
Gsmooth = medfilt2(Gmag, [5 5]);                % or use imgaussfilt(Gmag,1)
Mask = mat2gray(Gsmooth);                       % normalize mask to [0,1]
Mask = Mask.^0.7;                               % optional gamma to emphasize edges

% Mask the Laplacian and sharpen
alpha = 1.0;                                    % strength of masked Laplacian
L_masked = L .* Mask;
I_masked_sharp = I + alpha * L_masked;
I_masked_sharp = mat2gray(I_masked_sharp);

% Increase dynamic range (intensity transform)
I_final = imadjust(I_masked_sharp, stretchlim(I_masked_sharp, [0.01 0.99]), []);
% Alternatively, try local contrast (CLAHE):
I_final_clahe = adapthisteq(I_masked_sharp, 'ClipLimit', 0.02, 'Distribution', 'rayleigh');

%% Display results similar to the figure and the combined method
tiledlayout(2,4,'Padding','compact','TileSpacing','compact');

nexttile; imshow(I,[]);              title('(a) Original');
nexttile; imshow(L_disp,[]);         title('(b) Laplacian (display-scaled)');
nexttile; imshow(I_sharp,[]);        title('(c) Sharpened: I + Laplacian');
nexttile; imshow(Gmag_disp,[]);      title('(d) Sobel gradient |∇I|');

nexttile; imshow(Mask,[]);           title('Smoothed gradient mask');
nexttile; imshow(mat2gray(L_masked));title('Masked Laplacian (display)');
nexttile; imshow(I_masked_sharp,[]); title('Masked-Laplacian sharpened');
nexttile; imshow(I_final,[]);        title('Final (stretched)');

% Uncomment to compare with CLAHE
% figure; imshow(I_final_clahe,[]); title('Final with CLAHE');