function [d,imName] = get_cnn_resp(rootD,cnntype,imtype,top)
if nargin < 4
    top = 1;
end
subclass = dir([rootD '/stim/imagenet_subclass/*.csv']);
raw      = readtable([rootD '/rawdata/class/' cnntype '_' imtype '.csv'],'Delimiter',',');
if contains(cnntype,'diffusion')
    classId = readcell([rootD '/stim/codes/ImageNet_CoarseCat.csv']);
    tmppred  = classId(raw.predicted_index+2,2);
    raw.top1_class = tmppred;
    raw.image_path = raw.file_name;
end

[~,ord]  = sort(raw.image_path);
raw      = raw(ord,:);
d      = [];
imName = {};

if strcmp(imtype,'Sparse')
    pathName_remove = 'Cro_v05';
elseif strcmp(imtype,'Dense')
    pathName_remove = 'Cro_v07';
else
    pathName_remove = imtype;
end
for cls = 1:length(subclass)
    className  = subclass(cls).name(1:end-4);
    if ~contains(className,'_')
        childclass = readcell([subclass(cls).folder '/' subclass(cls).name],'Delimiter',',');
        corrans    = [childclass(:,2);className];
        corrans    = cellfun(@num2str,corrans,'UniformOutput',false);
        corrans    = strrep(corrans,'_',' ');
        if contains(cnntype,'diffusion')
            imIdx      = startsWith(raw.image_path,[className '_ILSVRC']);
        else
            imIdx      = contains(raw.image_path,['_' className '_ILSVRC']);
        end
        if top == 1
            output = cellfun(@(x) strrep(x,'_',' '),lower(raw.top1_class(imIdx)),'uniformoutput',false);
            d      = [d;ismember(output,corrans)];
        else
            output = cellfun(@(x) strrep(x,'_',' '),lower(raw.top5_class(imIdx)),'uniformoutput',false);
            d      = [d;contains(output,corrans)];
        end
        imName_raw = cellfun(@(x) strrep(x,[pathName_remove '_'],''),raw.image_path(imIdx), 'UniformOutput', false);
        imName     = [imName;imName_raw];
    end
end

end

