% Basic setup
clear; clc; close all;
I = imread('cameraman.tif');      % Grayscale demo image
figure; imshow(I); title('Original');

%% 1) Resize (scale)
I_res = imresize(I, 0.5, 'bicubic');   % 50% size
figure; imshow(I_res); title('Resized (0.5x)');

%% 2) Rotate (around image center, keep size)
I_rot = imrotate(I, 30, 'bilinear', 'crop');  % 30 deg CCW
figure; imshowpair(I, I_rot, 'montage'); title('Original | Rotated 30°');

%% 3) Translate (shift)
% Shift right by +30 px, up by -20 px; keep output size with gray fill
I_tr = imtranslate(I, [30, -20], 'OutputView', 'same', 'FillValues', 128);
figure; imshowpair(I, I_tr, 'montage'); title('Original | Translated [+30, -20]');

%% 4) Flip (reflection)
I_flip = fliplr(I);                        % Left-right mirror
figure; imshowpair(I, I_flip, 'montage'); title('Original | Flipped LR');

%% 5) General affine transform (rotate + shear + scale around center)
theta = deg2rad(20); shx = 0.3; sx = 1.1; sy = 0.9;  % params
cx = (size(I,2)+1)/2; cy = (size(I,1)+1)/2;          % image center

T_to0   = [1 0 0; 0 1 0; -cx -cy 1];                 % move center to origin
T_rot   = [cos(theta)  sin(theta)  0;                % rotation (row-vector form)
           -sin(theta) cos(theta)  0;
           0           0           1];
T_shear = [1 0 0; shx 1 0; 0 0 1];                    % x-shear
T_scale = [sx 0 0; 0 sy 0; 0 0 1];                    % scale
T_back  = [1 0 0; 0 1 0; cx cy 1];                    % move back

T_aff = T_to0 * T_rot * T_shear * T_scale * T_back;   % apply in this order
tform_aff = affine2d(T_aff);
I_aff = imwarp(I, tform_aff, 'OutputView', imref2d(size(I)), 'FillValues', 128);
figure; imshowpair(I, I_aff, 'montage'); title('Original | Affine (rot+shear+scale)');

