% % 
% % 
% % 

%%%%%%%%%%%%%%%%%%%%%%%%%% New normalize feature %%%%%%%%%%%%%%%%%%%
load('Pt1_sheuli_new_fea_0_180epoc_Norm.mat')
% 
X1=pt1_0_180epoc_sheuliNew_fea;

fea_index=[1:4,6:9,11];

X1=X1(:,:,:,fea_index);

ch_index=[1:19,21:61];

X1=X1(:,ch_index,:,:);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load Entropy Feature

load('P1_one_3entfea.mat')

% X=X_P1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 
X2=X_P1;

X_stat = cat(4,X1(1:180,1:60,:,:),X2(1:180,:,:,:));
% % 

X=X_stat;



% % 




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Feature selection %%%%%%%%%%%%%%%%%


%feature_selection

%FMI=[3,2,7,6,5,4,8,9,10,1,12,11] %original

FMI=[3,2,7,6,5,4,8,9,10,1];   %LGBM
% MBI=[1,2,4,5,6,5,3,8,7,9,10]

X=X(:,:,:,FMI);

% % 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% band selection %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% % SMI=[1,2,4,5,6,3,8,7,9,10]
SMI=[1,2,4,5,6,3,8,7,9]; %SVM
% 
% % 
X=X(:,:,SMI,:);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[TrainFeature, TestFeature, TrainLabel,TestLabel]=CrossvRabi(X);


csvwrite('Train.csv', TrainFeature);
csvwrite('Trainlabel.csv', TrainLabel);
csvwrite('Test.csv', TestFeature);
csvwrite('Testlabel.csv', TestLabel);

m=fitcsvm(TrainFeature, TrainLabel,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto','BoxConstraint',2);   %for four entropies we used .4 instead of 3. 
    
resultLabels = predict(m, TestFeature);
C = confusionmat(TestLabel, resultLabels);

[~,scores2] = resubPredict(m);
[x1,y1,~,auc1] = perfcurve(resultLabels,TestLabel,1);
    

PredictedResult= reshape(resultLabels, 60, 60); %channel and segments
imagesc(PredictedResult);

% M=mean(PredictedResult,2);
% 
% figure (2),
% plot(M);

R=PredictedResult'

for i=1:60
    z=find(R(:,i)==1);
    [v,id]=size(z);
    ContN(i)=v;
end
figure(2),
plot(ContN)


