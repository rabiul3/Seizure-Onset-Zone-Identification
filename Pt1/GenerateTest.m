function [TestFeature,TestLABEL]=GenerateTest(Test)
size(Test)

%Extract each band all features to create dimension for training set
Temp=[];
TestSample=[];
for bnd=1:size(Test,3)
  Temp=squeeze(Test(:,:,bnd,:));
  TestSample=cat(3,TestSample,Temp); %Train set generation
end

%Stack the feature from TrainSample
Temp1=[];
TestFeature=[];
for stck=1:size(TestSample,1)
  Temp1=squeeze(TestSample(stck,:,:));
  TestFeature=[TestFeature;Temp1];
end

%Generate test label
%Generate label
label=[];
for i=1:size(Test,2) 
  if i==10 || i==14 || i ==22   %%% 10,11,16
    label(i,:)=1;
  else
    label(i,:)=0; 
  end
end

%Training label
TestLABEL=[];
for i=1:size(Test,1)
  TestLABEL=[TestLABEL;label];
end

end

