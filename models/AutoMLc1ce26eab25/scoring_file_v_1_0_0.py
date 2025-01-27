# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.externals import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame({"Column1": pd.Series(["24186.0"], dtype="float64"), "age": pd.Series(["28.0"], dtype="float64"), "marital": pd.Series(["0.0"], dtype="float64"), "default": pd.Series(["0.0"], dtype="float64"), "housing": pd.Series(["0.0"], dtype="float64"), "loan": pd.Series(["1.0"], dtype="float64"), "month": pd.Series(["7.0"], dtype="float64"), "day_of_week": pd.Series(["1.0"], dtype="float64"), "duration": pd.Series(["101.0"], dtype="float64"), "campaign": pd.Series(["2.0"], dtype="float64"), "pdays": pd.Series(["999.0"], dtype="float64"), "previous": pd.Series(["2.0"], dtype="float64"), "poutcome": pd.Series(["0.0"], dtype="float64"), "emp.var.rate": pd.Series(["-1.7"], dtype="float64"), "cons.price.idx": pd.Series(["94.215"], dtype="float64"), "cons.conf.idx": pd.Series(["-40.3"], dtype="float64"), "euribor3m": pd.Series(["0.827"], dtype="float64"), "nr.employed": pd.Series(["4991.6"], dtype="float64"), "job_admin.": pd.Series(["0.0"], dtype="float64"), "job_blue-collar": pd.Series(["1.0"], dtype="float64"), "job_entrepreneur": pd.Series(["0.0"], dtype="float64"), "job_housemaid": pd.Series(["0.0"], dtype="float64"), "job_management": pd.Series(["0.0"], dtype="float64"), "job_retired": pd.Series(["0.0"], dtype="float64"), "job_self-employed": pd.Series(["0.0"], dtype="float64"), "job_services": pd.Series(["0.0"], dtype="float64"), "job_student": pd.Series(["0.0"], dtype="float64"), "job_technician": pd.Series(["0.0"], dtype="float64"), "job_unemployed": pd.Series(["0.0"], dtype="float64"), "job_unknown": pd.Series(["0.0"], dtype="float64"), "contact_cellular": pd.Series(["1.0"], dtype="float64"), "contact_telephone": pd.Series(["0.0"], dtype="float64"), "education_basic.4y": pd.Series(["0.0"], dtype="float64"), "education_basic.6y": pd.Series(["0.0"], dtype="float64"), "education_basic.9y": pd.Series(["0.0"], dtype="float64"), "education_high.school": pd.Series(["1.0"], dtype="float64"), "education_illiterate": pd.Series(["0.0"], dtype="float64"), "education_professional.course": pd.Series(["0.0"], dtype="float64"), "education_university.degree": pd.Series(["0.0"], dtype="float64"), "education_unknown": pd.Series(["0.0"], dtype="float64")})
output_sample = np.array([0])
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    try:
        model = joblib.load(model_path)
    except Exception as e:
        path = os.path.normpath(model_path)
        path_split = path.split(os.sep)
        log_server.update_custom_dimensions({'model_name': path_split[1], 'model_version': path_split[2]})
        logging_utilities.log_traceback(e, logger)
        raise


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
