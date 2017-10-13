import numpy as np
import pandas as pd



def Standardize_parameters(X):
    mean_vector = np.mean(X,axis = 0)
    std_vector = np.std(X,axis = 0)
    std_X = X - mean_vector/(std_vector)
    return std_X


def clean_data():
    """

    Output : X - Which is the feature matrix with no labels
              X_adj - The feature matrix with the 1's augmented to the right
              Y - Which is the labels

    """

    df_life = pd.read_csv('life_draft.csv')

    life_cont = ['MOVEMENT_COUNT', 'POLICY_AGE_ON_LAST_MOVEMENT',
                 'AGE_AT_ENTRY', 'RATE_CLASS', 'PREMIUM', 'COVER',
                 'DAY_OF_PAYMENT', 'PREMIUMS_RECEIVED', 'SUCCESSFUL_DEBITS',
                 'ATTEMPTED_DEBITS', 'DISTRIBUTION_CONTRACT',
                 'CALCULATED_INCOME_GROUP_CODE', 'AGE', 'OCCUPATIONAL_STATUS',
                 'RISK_GRADE','AGE_FIRST_PRODUCT',
                 'AGE_LATEST_PRODUCT', 'VALUE_SEGMENT', 'POLICY_AGE']

    life_int = ['GENDER','POLICY_STATUS','Accept_PremiumHoliday ', \
                'Reinstatement_Claim_Cancelled',\
                'Claim_Closed_Conversion', \
                'Lapse_Automatic_TermExpire_Premium',\
                'Divorced', 'BANK_RELATIONSHIP','Married',\
                'Other', 'ACCOUNT_TYPE','Separated', 'Single', 'UNKNOWN',\
                'Widow/Widower', 'LAPSED']

    df_life_cont =  df_life[life_cont]

    df_life_bi = df_life[life_int]


    life_cont =  df_life_cont.as_matrix()
    life_bi = df_life_bi.as_matrix()


    #rows = df_life.shape[0]
    cols = life_bi.shape[1]

    #Seperate the feature from the target
    Y  = life_bi[:,cols-1]
    X_bi  = life_bi[:,:cols-1]


    #convert array to vector
    Y = Y[:,np.newaxis]


    #horizontally stack ones from the
    X_cont = Standardize_parameters(life_cont)
    X =  np.hstack((X_cont,X_bi))
    X_adj = np.hstack((np.ones((X.shape[0],1)),X))

    return X,X_cont,Y


def data_split(X):
    """
    Input : matrix
    Outputs : 60,20 20 split
    """
    tot = X.shape[0]
    sixty = np.int(tot*0.6)
    twenty = np.int(tot*0.2)

    train = X[0:sixty,:]
    validation = X[sixty:sixty+twenty,:]
    test = X[sixty+twenty:sixty+(2*twenty),:]

    return train,validation,test


