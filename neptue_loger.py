
import neptune

run = neptune.init_run(
    project="astarteam/FinalProject",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhMDI5YzIxMy00NjE1LTQ2MDUtOTk3NS1jNDJhMjIzZDE0NDMifQ==",
)  # your credentials

run_info = {'model_name':'','dataset_name':'', }
run["info"] = run_info

params = {"learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params



run.stop()


import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


# Train and evaluate the classification model
for epoch in range(num_epochs):
    # Train the model and get predictions on the training set
    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    train_f1 = f1_score(y_train, train_preds, average='macro')
    train_precision = precision_score(y_train, train_preds, average='macro')
    train_recall = recall_score(y_train, train_preds, average='macro')

    # Log training metrics to Neptune
    run['train/accuracy'].log(train_acc)
    run['train/f1_score'].log(train_f1)
    run['train/precision'].log(train_precision)
    run['train/recall'].log(train_recall)

    # Get predictions on the validation set
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average='macro')
    val_precision = precision_score(y_val, val_preds, average='macro')
    val_recall = recall_score(y_val, val_preds, average='macro')
    val_roc_auc = roc_auc_score(y_val, val_preds)

    # Log validation metrics to Neptune
    run['validation/accuracy'].log(val_acc)
    run['validation/f1_score'].log(val_f1)
    run['validation/precision'].log(val_precision)
    run['validation/recall'].log(val_recall)
    run['validation/roc_auc'].log(val_roc_auc)

    # Log confusion matrix and classification report to Neptune
    cm = confusion_matrix(y_val, val_preds)
    class_report = classification_report(y_val, val_preds, digits=4, output_dict=True)

    run['validation/confusion_matrix'].log(neptune.types.File.as_html(cm))
    run['validation/classification_report'].log(neptune.types.File.as_html(class_report))

# End the Neptune experiment
run.stop()
