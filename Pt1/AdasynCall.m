function adasyn_featuresSyn=AdasynCall(FocalFeatue,nonFocalFeatue)

FOCALLABEL=[];
for i=1: size(FocalFeatue,1)
    FOCALLABEL(i,:)=1;
end

NONFOCALLABEL=[];
for i=1:size(nonFocalFeatue,1)
    NONFOCALLABEL(i,:)=0;
end

adasyn_features                 = [FocalFeatue; nonFocalFeatue];

adasyn_labels                   = [FOCALLABEL  ; NONFOCALLABEL  ];
adasyn_beta                     = [];   %let ADASYN choose default
adasyn_kDensity                 = [];   %let ADASYN choose default
adasyn_kSMOTE                   = [];   %let ADASYN choose default
% adasyn_featuresAreNormalized    = false;    %false lets ADASYN handle normalization
adasyn_featuresAreNormalized    = true;
[adasyn_featuresSyn, adasyn_labelsSyn] = ADASYN(adasyn_features,adasyn_labels, adasyn_beta, 6, 5, adasyn_featuresAreNormalized);  %%


%% For test
% % [adasyn_featuresSyn, adasyn_labelsSyn] = ADASYN(adasyn_features, adasyn_labels, adasyn_beta, 5, 5, adasyn_featuresAreNormalized);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% End of Oversampling %%%%%%%%%%%%%%%%%%%%%%
