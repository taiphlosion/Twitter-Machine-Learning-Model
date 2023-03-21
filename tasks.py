import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, classification_report
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean(text):
    text = text.lower() # to lowercase
    text = re.sub(r"@\S+", "", text) # remove mentions
    text = re.sub("http[s]?\://\S+","",text) # remove links
    text = re.sub(r"[0-9]", "", text) # remove numbers
    text = re.sub(r"won\'t", "will not", text) #checks for contractions
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"[$&+,:;=?@#|'<>.^*()%!-]", "", text) # remove special characters
    tokens = text.split()
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    filtered_text = ' '.join(filtered_tokens)

    return filtered_text

def bow_Classify(train, test):
    BOW = CountVectorizer()

    x_train = BOW.fit_transform(train['clean_text'])
    y_train = train['Sentiment']
    x_test = BOW.transform(test['Text'])
    y_test = test['Sentiment']

    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    lr_predicted = lr.predict(x_test)
    lr_pred_proba = lr.predict_proba(x_test)
    print('Bag of Words - Logistic Regression\n', classification_report(y_test, lr_predicted))

    svm = SVC(probability=True)
    svm.fit(x_train, y_train)
    svm_predicted = svm.predict(x_test)
    svm_pred_proba = svm.predict_proba(x_test)
    print('Bag of Words - SVM\n', classification_report(y_test, svm_predicted))

    NB = GaussianNB()
    NB.fit(x_train.toarray(), y_train)
    NB_predicted = NB.predict(x_test.toarray())
    NB_pred_proba = NB.predict_proba(x_test.toarray())
    print('Bag of Words - Bayes\n', classification_report(y_test, NB_predicted))

    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    rf_predicted = rf.predict(x_test)
    rf_pred_proba = rf.predict_proba(x_test)
    print('Bag of Words - Random Forest\n', classification_report(y_test, rf_predicted))

    models = ['Logistic Regression', 'Support Vector Machine', 'Naive Bayes Classifier', 'Random Forest Classifier']
    predictions = [lr_predicted, svm_predicted, NB_predicted, rf_predicted] #Makes two different array to store the values which is to be put in the confusion matrix
    pred_probabilities = [lr_pred_proba, svm_pred_proba, NB_pred_proba, rf_pred_proba]

    for model, prediction, pred_proba in zip(models, predictions, pred_probabilities): #For loops to plot out the matrix for each classification model
        disp = ConfusionMatrixDisplay(confusion_matrix(y_test.ravel(), prediction))
        disp.plot(
            include_values=True,
            cmap='gray',
            colorbar=False
        )
        disp.ax_.set_title(f"{model} Confusion Matrix")

    plt.figure(figsize=(30, 15)) #More labels to help understand the matrix
    plt.suptitle("ROC Curves")
    plot_index = 1

    for model, prediction, pred_proba in zip(models, predictions, pred_probabilities): #plots out the ROC curve to test model data
        fpr, tpr, thresholds = roc_curve(y_test, pred_proba[:, 1])
        auc_score = auc(fpr, tpr)
        plt.subplot(3, 2, plot_index)
        plt.plot(fpr, tpr, 'r', label='ROC curve')
        plt.title(f'Roc Curve - {model} - [AUC - {auc_score}]', fontsize=14)
        plt.xlabel('FPR', fontsize=12)
        plt.ylabel('TPR', fontsize=12)
        plt.legend()
        plot_index += 1
    plt.show()

def tfidf_Classify(train, test): 
    tf_idf = TfidfVectorizer()

    x_train = tf_idf.fit_transform(train['clean_text'])
    y_train = train['Sentiment']
    x_test = tf_idf.transform(test['Text'])
    y_test = test['Sentiment']

    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    lr_predicted = lr.predict(x_test)
    lr_pred_proba = lr.predict_proba(x_test)
    print('TF-IDF - Logistic Regression\n', classification_report(y_test, lr_predicted))


    svm = SVC(probability=True)
    svm.fit(x_train, y_train)
    svm_predicted = svm.predict(x_test)
    svm_pred_proba = svm.predict_proba(x_test)
    print('TF-IDF - SVM\n', classification_report(y_test, svm_predicted))

    NB = GaussianNB()
    NB.fit(x_train.toarray(), y_train)
    NB_predicted = NB.predict(x_test.toarray())
    NB_pred_proba = NB.predict_proba(x_test.toarray())
    print('TF-IDF - Bayes\n', classification_report(y_test, NB_predicted))

    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    rf_predicted = rf.predict(x_test)
    rf_pred_proba = rf.predict_proba(x_test)
    print('TF-IDF - Random Forest\n', classification_report(y_test, rf_predicted))

    models = ['Logistic Regression', 'Support Vector Machine', 'Naive Bayes Classifier', 'Random Forest Classifier']
    predictions = [lr_predicted, svm_predicted, NB_predicted, rf_predicted] #Makes two different array to store the values which is to be put in the confusion matrix
    pred_probabilities = [lr_pred_proba, svm_pred_proba, NB_pred_proba, rf_pred_proba]

    for model, prediction, pred_proba in zip(models, predictions, pred_probabilities): #For loops to plot out the matrix for each classification model
        disp = ConfusionMatrixDisplay(confusion_matrix(y_test.ravel(), prediction))
        disp.plot(
            include_values=True,
            cmap='gray',
            colorbar=False
        )
        disp.ax_.set_title(f"{model} Confusion Matrix")

    plt.figure(figsize=(30, 15)) #More labels to help understand the matrix
    plt.suptitle("ROC Curves")
    plot_index = 1

    for model, prediction, pred_proba in zip(models, predictions, pred_probabilities): #plots out the ROC curve to test model data
        fpr, tpr, thresholds = roc_curve(y_test, pred_proba[:, 1])
        auc_score = auc(fpr, tpr)
        plt.subplot(3, 2, plot_index)
        plt.plot(fpr, tpr, 'r', label='ROC curve')
        plt.title(f'Roc Curve - {model} - [AUC - {auc_score}]', fontsize=14)
        plt.xlabel('FPR', fontsize=12)
        plt.ylabel('TPR', fontsize=12)
        plt.legend()
        plot_index += 1
    plt.show()

    
