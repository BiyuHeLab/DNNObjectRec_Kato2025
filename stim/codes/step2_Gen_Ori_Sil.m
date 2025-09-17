clear;clc
rootD  = '/isilon/LFMI/VMdrive/Mugihiko/GlobalShape/Behav/DNNObjectRec_Kato2025/stim';
rawIms      = dir([rootD '/rawim/*.JPEG']);
savedir_Ori = [rootD '/genim/Ori'];
savedir_Sil = [rootD '/genim/Sil'];
if ~exist(savedir_Ori,'dir');mkdir(savedir_Ori);end
if ~exist(savedir_Sil,'dir');mkdir(savedir_Sil);end

imsize    = [256 256];
sil_color = 255*0.3;
bg_color  = 255*0.8;

for im = 1:length(rawIms)
    rawIm = imread([rawIms(im).folder '/' rawIms(im).name]);
    maskfn = strrep(rawIms(im).name,'.JPEG','.mat');
    load([rootD '/mask/' maskfn])

    % Adjusting image size
    xydiff = size(rawIm,1) - size(rawIm,2);
    if xydiff>0
        rawIm_pad = padarray(rawIm,[0 ceil(xydiff/2)],255);
        mask_pad = padarray(mask,[0 ceil(xydiff/2)],1);
    else
        rawIm_pad = padarray(rawIm,ceil(-xydiff/2),255);
        mask_pad = padarray(mask,ceil(-xydiff/2),1);
    end
    rawIm_rs = imresize(rawIm_pad,imsize);
    mask_rs  = repmat(imresize(mask_pad,imsize),1,1,3);

    % ori and sil
    sil           = uint8(mask_rs*bg_color); % to change background color, change here.
    sil(sil == 0) = sil_color;
    ori           = rawIm_rs;
    ori(mask_rs)  = bg_color;

    imwrite(ori,[savedir_Ori '/' strrep(rawIms(im).name,'.JPEG','_o.jpg')])
    imwrite(sil,[savedir_Sil '/' strrep(rawIms(im).name,'.JPEG','_s.jpg')])
end