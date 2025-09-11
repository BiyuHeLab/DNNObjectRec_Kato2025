%% updated
clear;clc
rootD = '/isilon/LFMI/VMdrive/Mugihiko/GlobalShape/Behav/GSB04';
addpath([rootD '/Script'])
savedir = [rootD '/analysis/F03_BS_accs_consist_scatter'];
if ~exist(savedir,'dir');mkdir(savedir);end
load([rootD '/rawdata/behav/summary.mat'])
load([rootD '/rawdata/class/BS_sum.mat'])
cnntypes   = {'resnet50','convnextL','cornet-s','vit_l_16',...
    'resnet50-sin','resnet50-blur-st','cornet-s-blur-st',...
    'clip_convnextL_image','clip_vit-l-laion_image',....
    'BS_convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384' 'BS_vit_large_patch14_clip_224:laion2b_ft_in12k_in1k'};
Cres = [49 130 189];
Ccon = [59 10 117];
Ccor = [0 178 0];
Cvit = [255 105 180];
Chum = [100 100 100];
col     = [Cres;Ccon;Ccor;Cvit;...
    Cres;Ccor;Cres;...
    Ccon;Cvit;...
    Ccon;Cvit]/255;
mark = {'o' 'o' 'o' 'o'...
    'v' '^' '^'...
    'diamond' 'diamond' 'square' 'square'};

xlims = [50 100;30 100;0 100;10 70;0 70;0 35;0 60];
ylims = [-0.2 1;0 0.5;0 0.7;0 0.7;0 0.7;0 0.6;0 0.7];
sz = 30;
imtype  = {'Ori' 'Fil' 'Lin' 'SilTex' 'Sil' 'Tex' 'LinFil'};
figlabel = {'' 'Fig.3a' 'Fig.3b' 'Fig.3c' 'Fig.3d' 'Fig.3e' 'Fig.3f'};

figW = 6.5;
figH = 6.5;
[~, colNo]= ismember(cnntypes,models);
colIdx= ismember(models,cnntypes);
clipIdx  = contains(models,'clip') &~colIdx;
hum_like_rank = cell(length(models),length(imtype));

for type = 1:length(imtype) % type = 3
    if strcmp(imtype{type},'Ori')
        hum = squeeze(d_indv(:,type,:));
    else
        hum = squeeze(d_batch_cl(:,type,:));
    end
    nbatch = size(hum,2);

    dat   = squeeze(mean(hum,1,'omitnan'));
    hum_accs  = mean(dat)*100;
    accs_sd = std(dat*100);
    kappa = nan(nchoosek(nbatch,2),1);
    ii = 1;
    for b1 = 1:nbatch
        for b2 = b1+1:nbatch
            hum1 = hum(:,b1);
            hum2 = hum(:,b2);
            excIdx = isnan(hum1)|isnan(hum2);
            hum1(excIdx) = [];
            hum2(excIdx) = [];

            hum1_av = mean(hum1);
            hum2_av = mean(hum2);

            c_exp = hum1_av*hum2_av + (1-hum1_av)*(1-hum2_av);
            c_obs = sum(hum1==hum2)/length(hum1);

            kappa(ii) = (c_obs - c_exp)/(1 - c_exp);
            ii = ii + 1;
        end
    end
    hum_kapp = mean(kappa);
    hum_kapp_min = min(kappa);
    hum_kapp_max = max(kappa);

    load([rootD '/analysis/BS01_error_consist/'  imtype{type} '.mat'],'models','kappa')

    figure
    hold on
    plot([xlims(type,1) xlims(type,2)],[hum_kapp hum_kapp],'--','Color',[.5 .5 .5 .5],'LineWidth',2);
    plot([hum_accs hum_accs],[ylims(type,1) ylims(type,2)],'--','Color',[.5 .5 .5 .5],'LineWidth',2);
    xregion(min(dat)*100,max(dat)*100,'FaceColor',[.8 .8 .8])
    yregion(hum_kapp_min,hum_kapp_max,'FaceColor',[.8 .8 .8])
    scatter(mean(accs(:,~colIdx,type),1,'omitnan')*100,mean(kappa(:,~colIdx),1,'omitnan'),sz,...
        'MarkerFaceColor',[.5 .5 .5],'MarkerFaceAlpha',0.6,'MarkerEdgeColor','none');
    axis square
    scatter(mean(accs(:,clipIdx,type),1,'omitnan')*100,mean(kappa(:,clipIdx),1,'omitnan'),sz,...
        'MarkerFaceColor','none','MarkerEdgeColor','r','LineWidth',1);
    for ii = 1:length(colNo)
        scatter(mean(accs(:,colNo(ii),type),1,'omitnan')*100,mean(kappa(:,colNo(ii)),1,'omitnan'),sz*2.5,mark{ii},...
            'MarkerFaceColor',col(ii,:),'MarkerFaceAlpha',0.85,'MarkerEdgeColor','none');
    end

    box off
    if type == 1
        xlabel('Accuracy (%)')
        ylabel('Consistensy')
    end
    ax = gca;
    ax.TickDir = 'out';
    ax.LineWidth = 1;
    ax.FontName = 'Helvetica';
    ax.FontSize = 6;
    ax.XTickLabelRotation = 0;
    ylabel('Error consistency')
    xlabel('Accuracy (%)')
    % title(imtype{type})

    [accsZ,accsM,accsS] = zscore(squeeze(mean(accs(:,:,type),1,'omitnan')));
    [kappZ,kappM,kappS] = zscore(squeeze(mean(kappa(:,1:end-1),1,'omitnan')));

    hum_accsZ = (hum_accs/100-accsM)/accsS;
    hum_kappZ = (hum_kapp-kappM)/kappS;
    accsD = hum_accsZ - accsZ;
    kappD = hum_kappZ - kappZ;

    D = sqrt(accsD.^2 + kappD.^2);

    [~,ord] = sort(D);
    hum_like_rank(:,type) = cellfun(@(x) strrep(x,'BS_',''),models(ord),'UniformOutput',false);
    rectangle('Position',[1 1 figW figH],'EdgeColor','none','FaceColor','none');

    set(gcf,'Color','white','Units', 'centimeters', 'Position', [1 1 figW figH], ...
        'PaperUnits', 'centimeters','defaultAxesXColor','k','defaultAxesYColor','k',...
        'defaultAxesZColor','k','PaperPosition', [0 0 figW figH], 'PaperSize',[figW figH])
    xlim(xlims(type,:))
    ylim(ylims(type,:))
    xticks(xlims(type,1):10:xlims(type,2))
    yticks(ylims(type,1):.1:ylims(type,2))
    % saveas(gcf,[savedir '/' imtype{type} '.fig'])
    % saveas(gcf,[savedir '/' imtype{type} '.png'])
    % exportgraphics(gcf, [savedir '/' imtype{type} '.pdf'], 'ContentType', 'vector');

    % source data
    if ~strcmp(figlabel{type},'')
        t = cell(sum(~colIdx)+1,3);
        t(2:end,1) = cellfun(@(x) strrep(x,'BS_',''),models(~colIdx),'UniformOutput',false);
        t{1,2} = "accuracy (%)";
        t{1,3} = 'error consistency';
        t(2:end,2) = num2cell(mean(accs(:,~colIdx,type),1,'omitnan')'*100);
        t(2:end,3) = num2cell(mean(kappa(:,~colIdx),1,'omitnan')');
        writetable(cell2table(t), [rootD '/sourcedat.xlsx'],'Sheet',figlabel{type},'WriteVariableNames',false);
    end
end

