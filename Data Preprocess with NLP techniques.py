import pandas as pd

# Reading DataSets
trainDataSet1 = pd.read_csv('AuroraIssues554.csv')
print("\nTraining Data Set 1\n")
print(trainDataSet1)
set1=trainDataSet1.iloc[:,1:-1].values

# Pre proccesing of DataSet1

# filter data in trainDataSet1
#This is used to filter out the relevant data column of Stroy
df1 =trainDataSet1.drop(index = trainDataSet1[trainDataSet1['issueType'] !='Story'].index)
df1.to_csv('Dataset1.csv',columns =['key','issueType','sprint','summary','description','storyPoint'])


# preprocess data in DataSet1
#This is used to pre process above data by removing missing value having data in the data feilds of description, stroypoint and summary
df2 =pd.read_csv('Dataset1.csv')
df3 =df2.fillna("No")
df4 =df3.drop(index = df3[df3['description'] =='No'].index)
df5 =df4.drop(index = df4[df4['storyPoint'] =='No'].index)
df6 =df5.drop(index = df5[df5['summary'] =='No'].index)
df6.to_csv('DataSet1PreprocessData.csv')


# remove punctuations in DataSet1
#This is used to remove the puctuations from the input data feilds of decription and summary and save those data in new data columns as clean_summary and clean_description
import string
def remove_punctuation(txt):
    txt_nonpunct="".join([c for c in txt if c not in string.punctuation])
    return txt_nonpunct
df6['clean_summary']=df6['summary'].apply(lambda x: remove_punctuation(x))
df6['clean_description']=df6['description'].apply(lambda x: remove_punctuation(x))
df6
df6.to_csv('DataSet1RemovePuctutation.csv')


# Tokanized data in DataSet1
#This is used to tokanized the data in the data feilds of clean_summary  and clean_description and include the tokinized data in the feilds of tokenized_summary and tokenized_description
import re
def tokenize(txt): 
    tokens = re.split('\W+',txt)
    return tokens
df6['tokenized_summary'] =df6['clean_summary'].apply(lambda x: tokenize(x.lower()))
df6['tokenized_description'] =df6['clean_description'].apply(lambda x: tokenize(x.lower()))
df6
df6.to_csv('DataSet1TokenizedData.csv')


# StopWords Removing in DataSet1
#This is used to remove stop words from tokenized_summary and tokenized_description. Remove stop words in english language is happened.
import nltk
stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(txt_tokenized):
    txt_clean =[word for word in txt_tokenized if word not in stopwords]
    return txt_clean

df6['removeSW_summary']= df6['tokenized_summary'].apply(lambda x: remove_stopwords(x))
df6['removeSW_description']= df6['tokenized_description'].apply(lambda x: remove_stopwords(x))
df6
df6.to_csv('DataSet1RemoveStopWords.csv')


# Creating Feature in DataSet1
#This is used to create one single data feild by combining two feilds of removeSW_summary and removeSW_description. And in here summary is followed by description order is happened. 
df6 = pd.read_csv('DataSet1RemoveStopWords.csv')
df6['Features']=df6['removeSW_summary']+' '+df6['removeSW_description']
df6
df6.to_csv('DataSet1FeatureCreation.csv')


