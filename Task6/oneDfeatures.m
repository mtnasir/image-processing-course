%% 1D Blob and 2nd Derivative of Gaussian (no offsets)
% Each blob (A,B,C) is a COLUMN.
% For each blob, we use 5 separate subplots (rows):
% 1) f(x)
% 2) n_sigma
% 3) d2 n_sigma / dx^2
% 4) (d2 n_sigma / dx^2) .* f(x)
% 5) CONVOLUTION: (sigma^2 * d2 n_sigma / dx^2) * f(x)

clear; clc;

%% Parameters
xMin = -6; xMax = 6;   N = 2000;
sigma = 1.2;

blobHalfWidths = [0.55, 1.25, 1.85];   % A,B,C half widths
smoothWin      = [41,   81,   101];    % smoothing windows for f(x)
titles         = {'Blob A','Blob B','Blob C'};

% Colors
clr.f    = [0.20 0.45 0.75];   % blue
clr.n    = [0.35 0.65 0.35];   % green
clr.d2n  = [0.10 0.10 0.10];   % black
clr.red  = [0.65 0.20 0.20];   % red/brown

%% Domain
x  = linspace(xMin, xMax, N).';
dx = x(2) - x(1);

%% Gaussian n_sigma and its 2nd derivative
n = (1/(sigma*sqrt(2*pi))) * exp(-(x.^2)/(2*sigma^2));                      % n_sigma
d2n = ((x.^2 - sigma^2)/(sigma^5*sqrt(2*pi))) .* exp(-(x.^2)/(2*sigma^2));  % d2 n_sigma / dx^2
kernel = (sigma^2) * d2n;                                                   % for convolution row

%% Build blobs f(x)
F = cell(1,3);
for i = 1:3
    f = double(abs(x) <= blobHalfWidths(i));   % rectangular blob
    f = smooth_boxcar(f, smoothWin(i));        % soften edges
    F{i} = f;
end

%% Plot: 5 rows x 3 columns
fig = figure('Color','w','Position',[80 60 1050 750]);
sgtitle('1D Blob and 2^{nd} Derivative of Gaussian','FontSize',22,'FontWeight','bold');

rowNames = { ...
    'f(x)', ...
    'n_\sigma', ...
    '\partial^2 n_\sigma/\partial x^2', ...
    '(\partial^2 n_\sigma/\partial x^2)\cdot f(x)', ...
    '( \sigma^2 \partial^2 n_\sigma/\partial x^2 ) * f(x)' ...
    };

for col = 1:3
    f = F{col};

    convResult = conv(f, kernel, 'same') * dx;  % convolution (continuous approx)

    Y = {
        f, ...
        n, ...
        d2n, ...
        d2n .* f, ...
        convResult ...
    };

    for row = 1:5
        ax = subplot(5,3,(row-1)*3 + col);
        hold(ax,'on'); box(ax,'on'); grid(ax,'off');

        y = Y{row};

        % Choose color by row (match reference style)
        if row == 1
            cc = clr.f; lw = 1.8;
        elseif row == 2
            cc = clr.n; lw = 1.6;
        elseif row == 3
            cc = clr.d2n; lw = 1.4;
        else
            cc = clr.red; lw = 1.6;
        end

        plot(ax, x, y, 'Color', cc, 'LineWidth', lw);

        % Cosmetics
        set(ax,'YTick',[]);
        xlim(ax, [xMin xMax]);

        % Only show x ticks on bottom row
        if row ~= 5
            set(ax,'XTick',[]);
        else
            xlabel(ax,'x');
        end

        % Column title on top row
        if row == 1
            title(ax, titles{col}, 'FontWeight','bold');
        end

        % Row label only on first column
        if col == 1
            ylabel(ax, rowNames{row}, 'Rotation',0, ...
                'HorizontalAlignment','right', 'VerticalAlignment','middle', ...
                'FontSize',12);
        end

        % Optional dashed center line like the image (only in Blob A column)
        if col == 1
            yl = ylim(ax);
            plot(ax, [0 0], yl, 'k--', 'LineWidth', 1.0);
        end
    end
end

%% -------- Local helper function --------
function y = smooth_boxcar(x, win)
    win = max(1, round(win));
    h = ones(win,1) / win;
    y = conv(x, h, 'same');
end