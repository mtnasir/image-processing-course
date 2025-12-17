clear
close all
% Morphological Operations in MATLAB
% Load an example binary image (you can replace this with your own image)
% originalImage = imread('text.png'); % Replace with your image file
% binaryImage=originalImage;

originalImage = imread('coins.png');
binaryImage = imbinarize(originalImage); % Directly binarize without rgb2gray

% binaryImage = imbinarize(rgb2gray(originalImage)); % Convert to binary if needed
% Define structuring element (you can modify this as needed)
se = strel('disk', 2); % Disk-shaped structuring element

% 1. Erosion
erodedImage = imerode(binaryImage, se);

% 2. Dilation
dilatedImage = imdilate(binaryImage, se);

% 3. Opening
openedImage = imopen(binaryImage, se);

% 4. Closing
closedImage = imclose(binaryImage, se);

% 5. Hit-or-Miss Transform
% Definestructuring elements for hit-or-miss
se1 = [1 1 1; 1 1 1; 0 0 0]; % Example pattern
se2 = [0 0 0; 0 0 0; 1 1 1];
hitMissResult = imerode(binaryImage, se1) & ~imdilate(binaryImage, se2);

% 6. Boundary Extraction
erodedForBoundary = imerode(binaryImage, se);
boundaryImage = binaryImage & ~erodedForBoundary;

% 7. Hole Filling
% Invert the image to make holes appear as objects
invertedImage = ~binaryImage;
% Seed point for hole filling (modify coordinates as needed)
seedPoint = [50, 50]; % [row, col]
% Perform hole filling using imfill
filledImage = imfill(invertedImage, seedPoint);
holeFilledImage = ~(filledImage);

% Display results
figure;
subplot(3, 3, 1); imshow(binaryImage); title('Original Binary Image');
subplot(3, 3, 2); imshow(erodedImage); title('Erosion');
subplot(3, 3, 3); imshow(dilatedImage); title('Dilation');
subplot(3, 3, 4); imshow(openedImage); title('Opening');
subplot(3, 3, 5); imshow(closedImage); title('Closing');
subplot(3, 3, 6); imshow(hitMissResult); title('Hit-or-Miss');
subplot(3, 3, 7); imshow(boundaryImage); title('Boundary Extraction');
subplot(3, 3, 8); imshow(holeFilledImage); title('Hole Filling');

% Note: For hit-or-miss transform, you might need to define custom structuring
% elements depending on the specific pattern you're trying to detect.