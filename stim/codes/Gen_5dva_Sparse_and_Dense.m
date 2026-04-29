clear;clc
rootD = '/isilon/LFMI/VMdrive/Mugihiko/GlobalShape/Behav/DNNObjectRec_Kato2025/stim';

dtype = {'Sparse' 'Dense'};
for d = 1:length(dtype)
    croimgs = dir([rootD '/genim/' dtype{d} '/*.jpg']);
    savedir = sprintf([rootD '/genim/' dtype{d} '5dva']);
    if ~exist(savedir,'dir')
        mkdir(savedir)
    end

    bg_color  = 255*0.8;
    for im = 1:length(croimgs)
        im0 = imread([croimgs(im).folder '/' croimgs(im).name]);
        im_rs = imresize(im0,0.5);
        im_pd = padarray(im_rs,[128/2 128/2],bg_color);
        imwrite(im_pd,[savedir '/' croimgs(im).name])
    end
end