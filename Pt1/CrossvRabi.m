function [TrainFeature, TestFeature, TrainLabel,TestLabel] = CrossvRabi(X)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
 Train=X(1:90,:,:,:);size(Train)
 Test=X(121:end,:,:,:);size(Test)
 
 %Extract each band all features to create dimension for training set
Temp=[];
TrainSample=[];
for bnd=1:size(Train,3)
  Temp=squeeze(Train(:,:,bnd,:));
  TrainSample=cat(3,TrainSample,Temp); %Train set generation
end

%Stack the feature from TrainSample
Temp1=[];
TrainFeature=[];
for stck=1:size(TrainSample,1)
  Temp1=squeeze(TrainSample(stck,:,:));
  TrainFeature=[TrainFeature;Temp1];
end

%Generate label
label=[];
for i=1:size(Train,2) 
  if i==10 || i==14 || i ==22    
    label(i,:)=1;
  else
    label(i,:)=0; 
  end
end

%Training label
TrainLABEL=[];
for i=1:size(Train,1)
  TrainLABEL=[TrainLABEL;label];
end

%Balancing the focal and nonfocal features
[TrainFeature,TrainLabel]=CallAdasyn(TrainFeature,TrainLABEL);


%%Process test features and label
[TestFeature,TestLabel]=GenerateTest(Test);
end

