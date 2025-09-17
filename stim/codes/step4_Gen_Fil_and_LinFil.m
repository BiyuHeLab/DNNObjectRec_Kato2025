clear;clc
rootD     = '/isilon/LFMI/VMdrive/Mugihiko/GlobalShape/Behav/DNNObjectRec_Kato2025/stim';
ori    = dir([rootD '/genim/Ori/*.jpg']);
savedir_linefil = [rootD '/genim/LinFil'];
saveidr_fil     = [rootD '/genim/Fil'];
if ~exist(savedir_linefil,'dir');mkdir(savedir_linefil);end
if ~exist(saveidr_fil,'dir');mkdir(saveidr_fil);end

bg_color  = 255*0.8;
imsize    = [256 256];

for im = 1:length(ori)
    ori0 = imread([ori(im).folder '/' ori(im).name]);
    sil0  = imread([strrep(ori(im).folder,'Ori','Sil') '/' strrep(ori(im).name,'o.jpg','s.jpg')]);
    line0 = imread([strrep(ori(im).folder,'Ori','Lin_raw') '/' ori(im).name]);

    % filled
    im_ms         = ~imbinarize(sil0);
    coveredO      = sum(im_ms,3)/3;
    coveredL      = sum(im_ms,3)/3;
    line0(~im_ms) = 0;
    ori0(~im_ms)  = 0;
    filledL       = line0;
    filledO       = ori0;
    while sum(coveredO == 0,'all') ~= 0
        tmp1 = randsample(size(line0,1),1);
        tmp2 = randsample(size(line0,1),1);
        ii   = randsample(4,1);
        if ii == 1
            imcL    = imcrop(line0,[tmp1 tmp2 imsize(2)-1 imsize(1)-1]);
            imcO    = imcrop(ori0,[tmp1 tmp2 imsize(2)-1 imsize(1)-1]);
            msc     = imcrop(im_ms,[tmp1 tmp2 imsize(2)-1 imsize(1)-1]);
            im_tmpL = padarray(imcL,[imsize(1) - size(imcL,1) imsize(2) - size(imcL,2)],'post');
            im_tmpO = padarray(imcO,[imsize(1) - size(imcO,1) imsize(2) - size(imcO,2)],'post');
            ms_tmp  = padarray(msc,[imsize(1) - size(msc,1) imsize(2) - size(msc,2)],'post');
        elseif ii == 2
            imcL    = imcrop(line0,[1 1 tmp1-1 tmp2-1]);
            imcO    = imcrop(ori0,[1 1 tmp1-1 tmp2-1]);
            msc     = imcrop(im_ms,[1 1 tmp1-1 tmp2-1]);
            im_tmpL = padarray(imcL,[imsize(1) - size(imcL,1) imsize(2) - size(imcL,2)],'pre');
            im_tmpO = padarray(imcO,[imsize(1) - size(imcO,1) imsize(2) - size(imcO,2)],'pre');
            ms_tmp  = padarray(msc,[imsize(1) - size(msc,1) imsize(2) - size(msc,2)],'pre');
        elseif ii == 3
            imcL    = imcrop(line0,[tmp1 1 imsize(1)-1 tmp2-1]);
            imcO    = imcrop(ori0,[tmp1 1 imsize(1)-1 tmp2-1]);
            msc     = imcrop(im_ms,[tmp1 1 imsize(1)-1 tmp2-1]);
            im_tmpL = padarray(imcL,[imsize(1) - size(imcL,1) 0],'pre');
            im_tmpL = padarray(im_tmpL,[0 imsize(2) - size(imcL,2)],'post');
            im_tmpO = padarray(imcO,[imsize(1) - size(imcO,1) 0],'pre');
            im_tmpO = padarray(im_tmpO,[0 imsize(2) - size(imcO,2)],'post');
            ms_tmp  = padarray(msc,[imsize(1) - size(msc,1) 0],'pre');
            ms_tmp  =  padarray(ms_tmp,[0 imsize(2) - size(imcL,2)],'post');
        else
            imcL    = imcrop(line0,[1 tmp2 tmp1-1 imsize(2)-1]);
            imcO    = imcrop(ori0,[1 tmp2 tmp1-1 imsize(2)-1]);
            msc     = imcrop(im_ms,[1 tmp2 tmp1-1 imsize(2)-1]);
            im_tmpL = padarray(imcL,[imsize(1) - size(imcL,1) 0],'post');
            im_tmpL = padarray(im_tmpL,[0 imsize(2) - size(imcL,2)],'pre');
            im_tmpO = padarray(imcO,[imsize(1) - size(imcO,1) 0],'post');
            im_tmpO = padarray(im_tmpO,[0 imsize(2) - size(imcO,2)],'pre');
            ms_tmp  = padarray(msc,[imsize(1) - size(msc,1) 0],'post');
            ms_tmp  = padarray(ms_tmp,[0 imsize(2) - size(imcL,2)],'pre');
        end
        im_tmp_sil = ms_tmp ~= 0;
        filledL     = filledL .* uint8(~im_tmp_sil);
        filledL     = filledL + im_tmpL;
        coveredL    = coveredL + sum(im_tmp_sil,3)/3;

        filledO     = filledO .* uint8(~im_tmp_sil);
        filledO     = filledO + im_tmpO;
        coveredO    = coveredO + sum(im_tmp_sil,3)/3;
    end

    imwrite(filledO,[saveidr_fil '/' strrep(ori(im).name,'o.jpg','f.jpg')])

    cont = imbinarize(filledL);
    filledL(~cont) = 0;
    filledL(cont) = bg_color;
    imwrite(filledL,[savedir_linefil '/' strrep(ori(im).name,'o.jpg','lf.jpg')])

end