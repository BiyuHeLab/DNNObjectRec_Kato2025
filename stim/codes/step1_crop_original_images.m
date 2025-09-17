clear;clc
rootD     = '/isilon/LFMI/VMdrive/Mugihiko/GlobalShape/Behav/DNNObjectRec_Kato2025/stim';
imD       = '/isilon/LFMI/VMdrive/Mugihiko/GlobalShape/ImageSet/ImageNet/ILSVRC/Data/CLS-LOC/val';
locD      = '/isilon/LFMI/VMdrive/Mugihiko/GlobalShape/ImageSet/ImageNet/LOC_val_solution.csv'; 
savedir   = [rootD '/rawim'];
if ~exist(savedir,'dir'); mkdir(savedir);end
val_loc = readtable(locD,'Delimiter',{' ','.',','});
targId  = readcell([rootD '/rawim/ImageNet_image_IDs.csv'],'Delimiter',',');

for im = 1:length(targId)
    im0 = imread([imD '/' targId{im,1}]);
    idx = find(strcmp(val_loc.ImageId,strrep(targId{im,1},'.JPEG','')));

    roi   = table2array(val_loc(idx,3:6));
    roi(:,3) = roi(:,3) - roi(:,1);
    roi(:,4) = roi(:,4) - roi(:,2);

    im0c = imcrop(im0,roi);
    imwrite(im0c,[savedir '/' targId{im,2} '_' targId{im,1}])
end
