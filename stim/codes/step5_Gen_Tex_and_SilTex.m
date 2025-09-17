
%{
Before running below, convert Fil images to texture using "texture_synthesis" available at:
https://github.com/LefdRida/texture_synthesis
The resulted images should be saved at 'stim/genim/Tex'

The only modification we made was target layers: change from 
layers = ["conv1_1", "pool1", "pool2", "pool3", "pool4"]
to 
layers = ["conv1_1","conv1_2", "pool1","conv2_1","conv2_2", "pool2"]
%}
clear;clc
rootD     = '/isilon/LFMI/VMdrive/Mugihiko/GlobalShape/Behav/DNNObjectRec_Kato2025/stim';

tex    = dir([rootD '/genim/Tex/*.jpg']);
savedir = [rootD '/genim/SilTex'];

if ~exist(savedir,'dir')
    mkdir(savedir)
end

for im = 1:length(tex)
    tex0 = imread([tex(im).folder '/' tex(im).name]);
    sil0 = imread([strrep(tex(im).folder,'Tex','Sil') '/' strrep(tex(im).name,'t.jpg','s.jpg')]);
    mask = imbinarize(sil0);

    tex0(mask) = 255*0.8;
    imwrite(tex0,[savedir '/' strrep(tex(im).name,'t.jpg','st.jpg')])
end
