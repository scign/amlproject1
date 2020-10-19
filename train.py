import argparse
import os
import joblib
import pandas as pd
import numpy as np
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, average_precision_score

def clean_data(data):
    # Dict for cleaning data
    months = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
    weekdays = {"mon":1, "tue":2, "wed":3, "thu":4, "fri":5, "sat":6, "sun":7}

    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    jobs = pd.get_dummies(x_df.job, prefix="job")
    x_df.drop("job", inplace=True, axis=1)
    x_df = x_df.join(jobs)
    x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
    x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
    x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
    x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)
    contact = pd.get_dummies(x_df.contact, prefix="contact")
    x_df.drop("contact", inplace=True, axis=1)
    x_df = x_df.join(contact)
    education = pd.get_dummies(x_df.education, prefix="education")
    x_df.drop("education", inplace=True, axis=1)
    x_df = x_df.join(education)
    x_df["month"] = x_df.month.map(months)
    x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
    x_df["poutcome"] = x_df.poutcome.apply(lambda s: 1 if s == "success" else 0)

    y_df = x_df.pop("y").apply(lambda s: 1 if s == "yes" else 0)
    return x_df, y_df
    
def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    parser.add_argument('--penalty', type=str, default='l1', help="Regression penalty")
    parser.add_argument('--solver', type=str, default='lbfgs', help="Solver algorithm")

    args = parser.parse_args()

    data_file_source = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
    ds = TabularDatasetFactory.from_delimited_files(path=data_file_source)
    x, y = clean_data(ds)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

    run = Run.get_context()

    run.log("Regularization Strength", np.float(args.C))
    run.log("Max iterations", np.int(args.max_iter))
    run.log("Penalty", args.penalty)
    run.log("Solver", args.solver)

    y_pred, y_score = None
    try:
        model = LogisticRegression(
            C=args.C,
            penalty=args.penalty,
            solver=args.solver,
            max_iter=args.max_iter
        ).fit(x_train, y_train)
    except ValueError as e:
        # catch incompatible parameters e.g. lbfgs doesn't support l1 penalty
        run.log("Error", str(e.with_traceback()))
        y_pred = np.zeroes_like(y_test)
        y_score = np.zeros_like(y_test, dtype=np.float)
    else:
        y_pred = model.predict(x_test)
        y_score = model.predict_proba(x_test)
        # save the model
        os.makedirs('outputs', exist_ok=True)
        joblib.dump(model, os.path.join('outputs','model.joblib'))
    finally:
        # let's calculate all the comparable AutoML metrics
        # so that we can properly compare this model to the AutoML batch
        auto_ml_classification_metrics = [
            'accuracy',
            'AUC_weighted',
            'average_precision_score_weighted',
            'norm_macro_recall',
            'precision_score_weighted'
        ]
        run.log("accuracy", accuracy_score(y_test, y_pred))
        run.log("AUC_weighted", roc_auc_score(y_test, y_score, average='weighted'))
        run.log("average_precision_score_weighted", average_precision_score(y_test, y_score, average='weighted'))
        run.log("norm_macro_recall", (recall_score(y_test, y_pred, average='macro')-0.5)/0.5)
        run.log("precision_score_weighted", precision_score(y_test, y_pred, average='weighted'))

if __name__ == '__main__':
    main()