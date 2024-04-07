function [ TrainFeature,TrainLabel] = CallAdasyn(TrainFeature,TrainLABEL)

%Extract focal feature from TrainLABEL
[Findex,v]=find(TrainLABEL==1);
FocalFeatue=TrainFeature(Findex,:);
%Extract non-focal Feature from MITrainFeature
[Nonindex,v]=find(TrainLABEL==0);
nonFocalFeatue =TrainFeature(Nonindex,:);

%Apply Adason to balance feature
adasyn_featuresSyn=AdasynCall(FocalFeatue,nonFocalFeatue);
%Combine adasyn_features and original focal features
Original_adasynTrain=[adasyn_featuresSyn; FocalFeatue];
%%Combine focal (contain original train focal feature and adasyn
%%features (Original_adasynTrain)) and non focal features (Ftrain2)
TrainFeature=[Original_adasynTrain;nonFocalFeatue];

%Create Train label (focal 2 and nonfocal 1)
TrainLabel=zeros(size( TrainFeature,1),1);
%TrainLabel(1:size(Original_adasynTrain,1)-1)=2;
TrainLabel(1:size(Original_adasynTrain,1))=1;

end

