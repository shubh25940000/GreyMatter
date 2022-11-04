import pandas as pd
"""The main function of this file is to clean up the file.
 That includes removing nulls, removing unimportant features, handling missing values etc."""

class clean_file:
    def __init__(self, df):
        self.df = df

    def check_nulls(self):
        nulls = pd.DataFrame(round(self.df.isna().sum()/len(self.df)*100, 2), columns= {'NaN'}).\
            reset_index(drop=False).rename(columns= {'index':'Columns'})
        nulls = nulls.sort_values(by='NaN', ascending=False)
        nulls = nulls[nulls['NaN'] > 0.0]
        return nulls

    def drop_nulls(self):
        nulls = self.check_nulls()
        if len(nulls) > 0:
            cols = list(nulls[nulls['NaN'] >= 15]['Columns'])
            df.drop(cols, inplace=True, axis = 1)
            return df

if __name__ == "__main__":
    df = pd.read_csv('/Users/shubhamchoudhury/Documents/Data Science/Machine learning/Self training/Life Expectancy Data.csv')
    E1 = clean_file(df)
    df = E1.drop_nulls()
    df.to_csv('/Users/shubhamchoudhury/Documents/Data Science/Machine learning/Self training/Removed_Nulls.csv', index = False)










