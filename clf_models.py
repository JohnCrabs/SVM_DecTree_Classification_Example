from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

from io_clf import *


def float_to_string_format(num):
    return "%.3f" % num


def make_predictions_and_error_estimation_printing(clf, x_train, x_test, y_train, y_test, cls_name="Classifier:"):
    # Make Predictions
    y_pred_train = clf.predict(x_train)
    y_pred_test = clf.predict(x_test)
    y_pred_proba_train = clf.predict_proba(x_train)
    y_pred_proba_test = clf.predict_proba(x_test)
    # Error Estimation
    acc_train = metrics.accuracy_score(y_train, y_pred_train)
    acc_test = metrics.accuracy_score(y_test, y_pred_test)
    pre_train = metrics.precision_score(y_train, y_pred_train, average='macro')
    pre_test = metrics.precision_score(y_test, y_pred_test, average='macro')
    rec_train = metrics.recall_score(y_train, y_pred_train, average='macro')
    rec_test = metrics.recall_score(y_test, y_pred_test, average='macro')
    f1_train = metrics.f1_score(y_train, y_pred_train, average='macro')
    f1_test = metrics.f1_score(y_test, y_pred_test, average='macro')
    aucroc_train_ovo = metrics.roc_auc_score(y_train, y_pred_proba_train, multi_class='ovo')
    aucroc_test_ovo = metrics.roc_auc_score(y_test, y_pred_proba_test, multi_class='ovo')
    aucroc_train_ovr = metrics.roc_auc_score(y_train, y_pred_proba_train, multi_class='ovr')
    aucroc_test_ovr = metrics.roc_auc_score(y_test, y_pred_proba_test, multi_class='ovr')
    # Print Errors
    print(cls_name)
    print("acc_train =", float_to_string_format(acc_train))
    print("acc_test =", float_to_string_format(acc_test))
    print("pre_train =", float_to_string_format(pre_train))
    print("pre_test =", float_to_string_format(pre_test))
    print("rec_train =", float_to_string_format(rec_train))
    print("rec_test =", float_to_string_format(rec_test))
    print("f1_train =", float_to_string_format(f1_train))
    print("f1_test =", float_to_string_format(f1_test))
    print("aucroc_train_ovo =", float_to_string_format(aucroc_train_ovo))
    print("aucroc_test_ovo =", float_to_string_format(aucroc_test_ovo))
    print("aucroc_train_ovr =", float_to_string_format(aucroc_train_ovr))
    print("aucroc_test_ovr =", float_to_string_format(aucroc_test_ovr))
    print()


def clf_svm_SVC_model(input_array, output_array):
    # Split and Scale Data
    x_train, x_test, y_train, y_test = train_test_split(input_array, output_array, train_size=0.2)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Create, Train and Export Classifier
    clf = svm.SVC(probability=True)
    clf.fit(x_train, y_train)
    save_clf("clf/svm_SVC/clf_svm_SVC", clf)

    # Make Predictions and Estimate Classsifier
    make_predictions_and_error_estimation_printing(clf, x_train, x_test, y_train, y_test, "SVM Classification:")


def clf_dec_tree_model(input_array, output_array):
    x_train, x_test, y_train, y_test = train_test_split(input_array, output_array, train_size=0.2)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    save_clf("clf/dec_tree/dec_tree", clf)

    # Make Predictions and Estimate Classsifier
    make_predictions_and_error_estimation_printing(clf, x_train, x_test, y_train, y_test,
                                                   "Decision Tree Classification:")
