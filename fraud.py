import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn import metrics
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def Train_model(data):
    y = data['Class']
    X = data.drop(columns=['Class'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=69, stratify=y)

    smote = SMOTE(random_state=69)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    model = GradientBoostingClassifier(random_state=69)
    model.fit(X_resampled, y_resampled)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    display = metrics.ConfusionMatrixDisplay(confusion_matrix= conf_matrix, display_labels=[0,1])
    display.plot()
    plt.show()
    return conf_matrix


df =Train_model(load_data("creditcard.csv"))
print(df)