datasetStream = cell(301,1);

for datasetIndex = 1 : 301

    datasetName = strcat('C:/Users/Rafael/Dropbox/PhD Code/Classifier Domains of Competence/S2/D',int2str(datasetIndex),'-tst.arff');
    wekaObj = loadARFF(datasetName);
    [mdata,featureNames,targetNDX,stringVals,relationName] = weka2matlab(wekaObj,[]);
    
    [row col] = size(mdata);
    
    datasetPRTOOLS = dataset(mdata(:,1:col-1),mdata(:,col));
    datasetStream(datasetIndex,1) = datasetPRTOOLS;
    
end;

