
%{
Before running below, convert Ori images to line drawings using "informative-drawings" available at:
https://github.com/carolineec/informative-drawings?tab=readme-ov-file

The resulted images should be saved at 'stim/genim/Lin_raw'
The "Style" option was set to "anime".
%}

clear;clc
rootD     = '/isilon/LFMI/VMdrive/Mugihiko/GlobalShape/Behav/DNNObjectRec_Kato2025/stim';

line    = dir([rootD '/genim/Lin_raw/*.jpg']);
savedir = [rootD '/genim/Lin'];

if ~exist(savedir,'dir')
    mkdir(savedir)
end

for im = 1:length(line)
    line0 = imread([line(im).folder '/' line(im).name]);
    sil0  = imread([strrep(line(im).folder,'Lin_raw','Sil') '/' strrep(line(im).name,'o.jpg','s.jpg')]);
    cont = imbinarize(line0);

    sil0(~cont) = 0;
    imwrite(sil0,[savedir '/' strrep(line(im).name,'o.jpg','l.jpg')])
end