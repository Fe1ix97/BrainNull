import os
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split




def generate_plot(data):
    data = data[['Group', 'M/F', 'Age', 'EDUC', 'SES',
                 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']]
    data.rename(columns={'M/F': 'Gender'}, inplace=True)
    data['SES'] = data['SES'].fillna(2.0)
    # Binary encode object columns
    data['Group'] = data['Group'].apply(lambda x: 1 if x == 'Demented' else 0)
    data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'M' else 0)
    data = data.astype('float64')

    corr = data.corr()
    plt.figure(figsize=(12, 6))
    sns.heatmap(corr, annot=True, vmin=-1)
    #plt.show()
    plt.savefig("images/heatmap.png")

    #Relazione tra sesso e demenza
    demented_group = data[data['Group'] == 1]['Gender'].value_counts()
    demented_group = pd.DataFrame(demented_group)
    demented_group.index = ['Male', 'Female']
    demented_group.plot(kind='bar', figsize=(8, 6))
    plt.title('Gender vs Dementia', size=16)
    plt.xlabel('Gender', size=14)
    plt.ylabel('Patients with Dementia', size=14)
    plt.xticks(rotation=0)
    #plt.show()
    plt.savefig("images/general1.png")

    #Relazione fra Età e Normalized Whole Brain Volume
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Age', y='nWBV', data=data, hue='Group')
    plt.title('Age vs Normalized Whole Brain Volume', size=16)
    plt.xlabel('Age', size=14)
    plt.ylabel('Normalized Whole Brain Volume', size=14)
    #plt.show()
    plt.savefig("images/general4.png")

    #Distribuzione del punteggio MMSE nelle persone affette da demenza ssenile e non
    plt.figure(figsize=(10, 6))
    sns.kdeplot(x='MMSE', shade=True, hue='Group', data=data)
    plt.title('Distrubtion of MMSE scores in Demented and Nondemented Patients', size=16)
    plt.xlim(data['MMSE'].min(), data['MMSE'].max())
    plt.xlabel('MMSE Score', size=14)
    plt.ylabel('Density of Scores', size=14)
    #plt.show()
    plt.savefig("images/general2.png")

    #Relazione tra gli anni di studio e la demenza senile
    plt.figure(figsize=(10, 6))
    sns.kdeplot(x='EDUC', shade=True, hue='Group', data=data)
    plt.title('Anni di studio VS Alzheimer', size=16)
    plt.xlabel('Education (years)', size=14)
    plt.ylabel('Density', size=14)
    #plt.show()
    plt.savefig("images/general3.png")

def loadDataSet(genplot):
    for dirname, _, filenames in os.walk('/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    print("Caricamento dataset")
    data = pd.read_csv('input/oasis_longitudinal.csv')
    if genplot:
        generate_plot(data)

    data['M/F'] = [1 if each == "M" else 0 for each in data['M/F']]
    data['Group'] = [1 if each == "Demented" or each == "Converted" else 0 for each in data['Group']]

    median = data['MMSE'].median()
    data['MMSE'].fillna(median, inplace=True)
    median = data['SES'].median()
    data['SES'].fillna(median, inplace=True)

    y = data['Group'].values
    X = data[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]



    correlation_matrix = data.corr()
    data_corr = correlation_matrix['Group'].sort_values(ascending=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=25)
    return X_train, X_test, y_train, y_test, X, y

# Mi aiuta a dividere le colonne di oggetti da quelle numeriche
def sepdatatype(data):
    categorical_data = pd.DataFrame()
    numerical_data = pd.DataFrame()

    for col in data.columns:
        if data[col].dtype == 'O':
            categorical_data = pd.concat([categorical_data, pd.DataFrame(data[col])], axis=1)
        else:
            numerical_data = pd.concat([numerical_data, pd.DataFrame(data[col])], axis=1)
    return categorical_data, numerical_data

# Questa funzione è stata pensata per rimuovere i valori anomali da una colonna in modo da ottenere un set di dati ridotto
def olrem(data):
    length = data.shape[0]
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lrange = Q1 - 1.5 * IQR
    urange = Q3 + 1.5 * IQR
    for i in range(length):

        if data[i] < lrange or data[i] > urange:
            data[i] = 0

    return data


def repol(data):
    length = data.shape[0]
    for col in data.columns:

        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lrange = Q1 - 1.5 * IQR
        urange = Q3 + 1.5 * IQR

        mn = olrem(data[col]).mean()
        for j in range(length):
            if data[col][j] < lrange or data[col][j] > urange:
                data[col][j] = mn
    return data