import pickle
import warnings
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from MRI.KNN import knnClassifier, kncWithGridView
from MRI.RandomForestClassifer import rfClassifier, rfcWithGridView
from MRI.decision_tree import dtClassifier, dtcWithGridView
from MRI.logisticRegression import log_reg, log_regWithGridView
from MRI.mpl import mlpClassifier, mlpWithGridView
from utils.utils import loadDataSet

warnings.filterwarnings("ignore")

dct_filename = 'models/dct_pkl.pkl'
knc_filename = 'models/knc_model.pkl'
rfc_filename = 'models/rfc_model.pkl'
mlp_filename = 'models/mlp_model.pkl'
lr_filename = 'models/lr_model.pkl'

genplot = True #impostare su True per generare e salvare i grafici
X_train, X_test, y_train, y_test, x, y = loadDataSet(genplot)
df_ytrain = pd.DataFrame(y_train)
df_ytest = pd.DataFrame(y_test)
scaler = StandardScaler().fit(X_train)
# scaler = MinMaxScaler().fit(X_trainval)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)




def executeClassifier():
    dct, dct_fpr, dct_tpr = dtClassifier(X_train_scaled, X_test_scaled, y_train, y_test, y, True, genplot)
    knn, knn_fpr, knn_tpr = knnClassifier(X_train_scaled, X_test_scaled, y_train, y_test, y, True, genplot)
    logr, lgr_fpr, lgr_tpr = log_reg(X_train_scaled, X_test_scaled, y_train, y_test, y, True, genplot)
    rfc, rfc_fpr, rfc_tpr = rfClassifier(X_train_scaled, X_test_scaled, y_train, y_test, y, True, genplot)
    mlp, mlp_frp, mlp_tpr = mlpClassifier(X_train_scaled, X_test_scaled, y_train, y_test, y, True, genplot)
    '''export dei classificatori addestrati'''

    pickle.dump(dct, open(dct_filename, 'wb'))
    pickle.dump(knn, open(knc_filename, 'wb'))
    pickle.dump(rfc, open(rfc_filename, 'wb'))
    pickle.dump(mlp, open(mlp_filename, 'wb'))
    pickle.dump(logr, open(lr_filename, 'wb'))

    if genplot:
        plt.figure(figsize=(5, 5), dpi=100)
        plt.plot(lgr_fpr, lgr_tpr, marker='.', label='Logistic Regression')
        plt.plot(rfc_fpr, rfc_tpr, linestyle=':', label='Random Forest')
        plt.plot(dct_fpr, dct_fpr, linestyle='-.', label='Decision Tree')
        plt.plot(knn_tpr, knn_fpr, linestyle='-.', label='K-Neighbors')
        plt.plot(mlp_frp, mlp_tpr, linestyle='-.', label='Multi Layer Perceptron')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        #plt.show()
        plt.savefig("images/comparazione.png")


def gridSearch():
    dtc_gd = dtcWithGridView(X_train_scaled, y_train)  # Decision Tree Classifer
    knn_gd = kncWithGridView(X_train_scaled, y_train)  # Multi Layer Perceptron
    rfc_gd = rfcWithGridView(X_train_scaled, y_train)  # K-Neighbors
    mlp_gd = mlpWithGridView(X_train_scaled, y_train)  # Multi Layer Perceptron
    lr_gd = log_regWithGridView(X_train_scaled, y_train)
    print("\nBEST PARAMS:")
    print("- DTC: ", dtc_gd.best_params_)
    print("- KNN: ", knn_gd.best_params_)
    print("- RFC: ", rfc_gd.best_params_)
    print("- MLP: ", mlp_gd.best_params_)
    print("- LR: ", lr_gd.best_params_)


def prediction(reconstructed_model=None):

    #dct_load = pickle.load(open(dct_filename, 'rb'))
    #knn_load = pickle.load(open(knc_filename, 'rb'))
    rfc_load = pickle.load(open(rfc_filename, 'rb'))
    mlp_load = pickle.load(open(mlp_filename, 'rb'))
    lr_load = pickle.load(open(lr_filename, 'rb'))

    print("Loading test data.. \n")
    # 'M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF'
    xx = [[0, 87, 14, 2, 27, 1987, 0.696, 0.883]]
    xy = [[1, 68, 6, 3, 29, 1968, 0.954, 1.123]]
    test = xy
    #xx_pred1 = dct_load.predict(test) #SCARSI RISULTATI NELLA FASE INIZIALE
    #xx_pred2 = knn_load.predict(test) #SCARSI RISULTATI NELLA FASE INIZIALE
    xx_pred3 = rfc_load.predict(test)

    xx_pred4 = mlp_load.predict(test)
    xx_pred5 = lr_load.predict(test)

    # PREDIZIONE3
    d = {'RaF': [xx_pred3[0]], 'MLP': [xx_pred4[0]], 'LR': [xx_pred5[0]]}
    prev = pd.DataFrame(data=d)

    demented = "demented" if prev.mode(1).loc[0].values[0]==1 else "nondemented"
    with pd.option_context('display.max_rows', 1, 'display.max_columns', 6):
        print(prev)
    print("\nThe most predicted disease is", demented, "\n\n")


print("Alzheimer prediction\n")

menu_options = {
    1:'Train IA',
    2:'Optimize parameters',
    3:'Run test',
    4:'Exit',
}


def print_menu():
    for key in menu_options.keys():
        print(key, '--', menu_options[key])

if __name__ == '__main__':
    while (True):
        print_menu()
        option = ''
        try:
            option = int(input('Enter your choice: '))
        except:
            print('Wrong input. Please enter a number ...')
        if option == 1:
            executeClassifier()
        elif option == 2:
            gridSearch()
        elif option == 3:
            prediction()
        elif option == 4:
            print('by Zingaro Felice\nMat.: 660972')
            exit()
        else:
            print('Invalid option. Please enter a number between 1 and 4.')
