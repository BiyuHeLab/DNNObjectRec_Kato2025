clear;clc
rootD = '/isilon/LFMI/VMdrive/Mugihiko/GlobalShape/Behav/DNNObjectRec_Kato2025/stim';
sil    = dir([rootD '/genim/Sil/*.jpg']);
dtype = {'Sparse' 'Dense'};

for d = 1:length(dtype)
    switch dtype{d}
        case 'Sparse'
            dotDiameter = 10;
            dotDensity  = 0.03;
        case 'Dense'
            dotDiameter = 10;
            dotDensity  = 0.0525;
    end
    savedir = sprintf([rootD '/genim/%s'],dtype{d});

    if ~exist(savedir,'dir')
        mkdir(savedir)
    end
    sil_color = 255*0.3;
    bg_color  = 255*0.8;
    for im = 1:length(sil)
        sil0 = imread([sil(im).folder '/' sil(im).name]);
        mask = imbinarize(sil0);
        mask = ~mask(:,:,1);

        [rows, cols] = size(mask);

        % Create a blank canvas for the triangle art
        triangleArtImage = zeros(rows, cols);

        % Calculate the total number of triangles based on density and silhouette area
        silhouetteArea = sum(mask(:));
        totalTriangles = round(silhouetteArea * dotDensity);

        % Generate random triangle positions within the silhouette
        [silhouetteY, silhouetteX] = find(mask);
        numSilhouettePixels = length(silhouetteY);

        % Randomly sample positions for triangles
        randomIndices = randi(numSilhouettePixels, totalTriangles, 1);
        sampledY = silhouetteY(randomIndices);
        sampledX = silhouetteX(randomIndices);

        % Create a triangle for each sampled position
        halfSide = dotDiameter / 2;
        [xGrid, yGrid] = meshgrid(1:cols, 1:rows);

        for i = 1:totalTriangles
            % Define the center of the plus sign
            centerX = sampledX(i);
            centerY = sampledY(i);

            % Draw a "+" shape (cross)
            thickness = round(dotDiameter / 2); % Half length of the cross arms
            for t = -thickness:thickness
                % Horizontal line
                if centerX + t > 0 && centerX + t <= cols
                    triangleArtImage(centerY, centerX + t) = 1;
                end
                % Vertical line
                if centerY + t > 0 && centerY + t <= rows
                    triangleArtImage(centerY + t, centerX) = 1;
                end
            end
        end

        dot           = uint8(~triangleArtImage*bg_color); % to change background color, change here.
        dot(dot == 0) = sil_color;

        imwrite(dot,[savedir '/' strrep(sil(im).name,'_s.jpg','_d.jpg')])
    end
end