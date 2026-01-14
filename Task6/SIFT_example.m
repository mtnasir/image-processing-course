% SIFT Feature Detection and Matching
% Read the images
clear
close all
clc
img1 = imread('IMG_1187.jpg');  % Reference image
img2 = imread('IMG_1188.jpg');       % Test image to compare
% img1=imrotate(img1,-90);
% Convert to grayscale if needed
if size(img1, 3) == 3
    gray1 = rgb2gray(img1);
else
    gray1 = img1;
end

if size(img2, 3) == 3
    gray2 = rgb2gray(img2);
else
    gray2 = img2;
end

% Detect SIFT features
points1 = detectSIFTFeatures(gray1);
points2 = detectSIFTFeatures(gray2);

fprintf('SIFT features detected in reference image: %d\n', points1.Count);
fprintf('SIFT features detected in test image: %d\n', points2.Count);

% Extract SIFT descriptors
[features1, valid_points1] = extractFeatures(gray1, points1);
[features2, valid_points2] = extractFeatures(gray2, points2);

% Match features between the two images
indexPairs = matchFeatures(features1, features2, 'MaxRatio', 0.8);

% Retrieve matched points
matchedPoints1 = valid_points1(indexPairs(:, 1), :);
matchedPoints2 = valid_points2(indexPairs(:, 2), :);

fprintf('Number of matched features: %d\n', size(indexPairs, 1));

% Visualize matched features
figure;
showMatchedFeatures(img1, img2, matchedPoints1, matchedPoints2, 'montage');
title(sprintf('SIFT Feature Matching - %d matches found', size(indexPairs, 1)));
legend('Matched points in reference image', 'Matched points in test image');

% Display individual SIFT features
figure;
subplot(1, 2, 1);
imshow(img1);
hold on;
plot(points1.selectStrongest(50));
title('Top 50 SIFT Features - Reference Image');
hold off;

subplot(1, 2, 2);
imshow(img2);
hold on;
plot(points2.selectStrongest(50));
title('Top 50 SIFT Features - Test Image');
hold off;

% Optional: Estimate geometric transformation
if size(indexPairs, 1) >= 4
    [tform, inlierIdx] = estimateGeometricTransform2D(...
        matchedPoints1, matchedPoints2, 'affine');
    
    inlierPoints1 = matchedPoints1(inlierIdx, :);
    inlierPoints2 = matchedPoints2(inlierIdx, :);
    
    fprintf('Number of inlier matches: %d\n', sum(inlierIdx));
    
    % Visualize inlier matches
    figure;
    showMatchedFeatures(img1, img2, inlierPoints1, inlierPoints2, 'montage');
    title(sprintf('Inlier Matches - %d inliers', sum(inlierIdx)));
end

% Calculate matching score (percentage of matches)
matching_score = size(indexPairs, 1) / min(points1.Count, points2.Count) * 100;
fprintf('Matching score: %.2f%%\n', matching_score);