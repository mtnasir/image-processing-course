%% Basic MATLAB Image Segmentation Examples
%% Using only built-in Image Processing Toolbox commands

% Load an image
img = imread('cameraman.tif');  % Use any image you have

% Convert to grayscale if needed
if size(img, 3) == 3
    gray = rgb2gray(img);
else
    gray = img;
end

%% 1. SIMPLE THRESHOLDING
% Otsu method
level = graythresh(gray);
binary = imbinarize(gray, level);

% Manual threshold
manual_thresh = gray > 128;

% Adaptive thresholding
adaptive = imbinarize(gray, 'adaptive', 'ForegroundPolarity', 'dark');

%% 2. EDGE DETECTION
% Sobel edges
sobel_edges = edge(gray, 'sobel');

% Canny edges
canny_edges = edge(gray, 'canny');

%% 3. WATERSHED SEGMENTATION
% Simple watershed segmentation
bw = imbinarize(gray);
% Compute gradient magnitude for watershed
[Gx, Gy] = imgradientxy(gray, 'sobel');
grad_mag = sqrt(Gx.^2 + Gy.^2);
result = watershed(grad_mag);

%% 4. K-MEANS CLUSTERING
% Reshape image to pixel list
pixels = double(reshape(gray, [], 1));
[idx, centers] = kmeans(pixels, 3);
segmented = reshape(uint8(centers(idx)), size(gray));

%% 5. HOUGH LINE DETECTION
% Find lines in edge image
[H, theta, rho] = hough(canny_edges);
peaks = houghpeaks(H, 5);
lines = houghlines(canny_edges, theta, rho, peaks);

%% DISPLAY RESULTS
figure;
subplot(2,3,1); imshow(gray); title('Original');
subplot(2,3,2); imshow(binary); title('Otsu Threshold');
subplot(2,3,3); imshow(sobel_edges); title('Sobel Edges');
subplot(2,3,4); imshow(canny_edges); title('Canny Edges');
subplot(2,3,5); imshow(segmented, []); title('K-means Segmentation');
subplot(2,3,6); imshow(result, []); title('Watershed');

%% DRAW DETECTED LINES
if ~isempty(lines)
    figure;
    imshow(gray); hold on;
    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'green');
    end
    title('Detected Lines');
end