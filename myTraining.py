import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

def split(data,ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data)*ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

if __name__ == "__main__":
    df=pd.read_csv(r"C:\Users\Priyanshu Gupta\Desktop\final\data.csv")
    train,test=split(df,0.2)
    X_train=train.iloc[:,0:6].values
    X_test=test.iloc[:,0:6].values
    Y_train=train.iloc[:,6].values.reshape(2000,)
    Y_test=test.iloc[:,6].values.reshape(499,)
    clf=LogisticRegression()
    clf.fit(X_train,Y_train)
    
    #open a file, where you want to store the data
    file=open("model.pkl", 'wb')

    # dump information to that file
    pickle.dump(clf,file)
    file.close()
    
