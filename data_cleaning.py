import pandas as pd
import os


# prepare the dataset
df = pd.read_csv('data/baoxianzhidao_filter.csv')

# question and title are redudant
df.drop(columns=['question'], inplace=True)

# drop the row when nan value in reply colums
df.dropna(how="any", axis=0, subset=["reply"], inplace=True)

# drop all the wrose answer
all0 = df[ df['is_best'] == 0 ].index
df.drop(all0 , inplace=True)

# drop is_best colums
df.drop(['is_best'], axis=1,inplace=True)

#rename the colums name
df_new = df.rename(columns={'title': 'question','reply': 'answer'})

# write to the new csv file for later
df_new.to_csv('data/insurance_qa.csv',index=False)



data = pd.read_csv ("data/chatterbot.tsv", sep = '\t',header=None)
names=['question','answer']
data.columns=names
data.to_csv('data/chatterbot.csv',index=False)
