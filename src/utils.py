# baseline model with 25 features 
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def train_validation_metrics(data_splits: tuple, model, LGB: bool): 

    X_train, X_val, X_test, y_train, y_val, y_test = data_splits

    lr = model

    if LGB: 

        lr.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    else:

        lr.fit(X_train, y_train)

    y_train_preds = lr.predict(X_train)

    y_train_prob = lr.predict_proba(X_train)[:, 1]

    y_val_preds = lr.predict(X_val)
    y_val_prob = lr.predict_proba(X_val)[:, 1]


    print('Training Metrics')

    training_f1 = f1_score(y_train, y_train_preds, average='weighted')
    print(f'F1 Score:{training_f1}')

    print(f'Recall score: {recall_score(y_train, y_train_preds, pos_label="Charged Off")}')

    print(f'Precision score: {precision_score(y_train, y_train_preds, pos_label="Charged Off")}')

    cm_train = confusion_matrix(y_train, y_train_preds, labels=lr.classes_[::-1])

    print(f'{cm_train}')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_train,
                              display_labels=lr.classes_[::-1])


    print('--' * 30)

    # Validation metrics 

    print('Test Metrics')
    val_f1 = f1_score(y_val, y_val_preds, average='weighted')
    print(f'F1 Score:{val_f1}')

    print(f'Recall score: {recall_score(y_val, y_val_preds, pos_label="Charged Off")}')

    print(f'Precision score: {precision_score(y_val, y_val_preds, pos_label="Charged Off")}')

    cm = confusion_matrix(y_val, y_val_preds, labels=lr.classes_[::-1])

    print(f'{cm}')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=lr.classes_[::-1])
    
    disp.plot()

    return training_f1, val_f1
