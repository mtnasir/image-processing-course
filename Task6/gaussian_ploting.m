% Gaussian function and its derivatives
% Define the Gaussian function
gaussian = @(x, sigma) exp(-x.^2 ./ (2 * sigma^2)) ./ (sigma * sqrt(2 * pi));

% Define the first derivative of Gaussian
gaussian_derivative1 = @(x, sigma) -x .* exp(-x.^2 ./ (2 * sigma^2)) ./ (sigma^3 * sqrt(2 * pi));

% Define the second derivative of Gaussian
gaussian_derivative2 = @(x, sigma) (x.^2 - sigma^2) .* exp(-x.^2 ./ (2 * sigma^2)) ./ (sigma^5 * sqrt(2 * pi));

% Parameters
sigma = 1;  % Standard deviation
x = linspace(-5, 5, 1000);  % x range

% Calculate functions
y1 = gaussian(x, sigma);
y2 = gaussian_derivative1(x, sigma);
y3 = gaussian_derivative2(x, sigma);

% Create the plot
figure;
hold on;
grid on;

% Plot Gaussian function
plot(x, y1, 'b-', 'LineWidth', 2, 'DisplayName', 'Gaussian Function');

% Plot first derivative
plot(x, y2, 'r-', 'LineWidth', 2, 'DisplayName', 'First Derivative');

% Plot second derivative
plot(x, y3, 'g-', 'LineWidth', 2, 'DisplayName', 'Second Derivative');

% Customize the plot
xlabel('x', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('Gaussian Function and Its Derivatives', 'FontSize', 14);
legend('Location', 'best');
set(gca, 'FontSize', 11);

% Add text annotations
text(-4, 0.35, sprintf('\\sigma = %.1f', sigma), 'FontSize', 10, ...
     'BackgroundColor', 'white', 'EdgeColor', 'black');

hold off;

% Display mathematical expressions in command window
fprintf('\nMathematical expressions:\n');
fprintf('Gaussian function: f(x) = (1/(\\sigma\\sqrt{2\\pi})) * exp(-x^2/(2\\sigma^2))\n');
fprintf('First derivative: f''(x) = (-x/(\\sigma^3\\sqrt{2\\pi})) * exp(-x^2/(2\\sigma^2))\n');
fprintf('Second derivative: f''''(x) = ((x^2-\\sigma^2)/(\\sigma^5\\sqrt{2\\pi})) * exp(-x^2/(2\\sigma^2))\n\n');