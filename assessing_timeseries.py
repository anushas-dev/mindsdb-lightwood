import pandas as pd
from lightwood.api.high_level import (
    ProblemDefinition,
    json_ai_from_problem,
    code_from_json_ai,
    predictor_from_code,
)

# Load a pandas dataset
df = pd.read_csv(
    "Train.csv"
)

# Define the prediction task by naming the target column
pdef = ProblemDefinition.from_dict(
    {
        "target": "feature_3",  # column you want to predict
    }
)

# Generate JSON-AI code to model the problem
json_ai = json_ai_from_problem(df, problem_definition=pdef)

# OPTIONAL - see the JSON-AI syntax
#print(json_ai.to_json())

# Generate python code
code = code_from_json_ai(json_ai)

# OPTIONAL - see generated code
#print(code)

# Create a predictor from python code
predictor = predictor_from_code(code)

# Train a model end-to-end from raw data to a finalized predictor
predictor.learn(df)

# Make the train/test splits and show predictions for a few examples
test_df = predictor.split(predictor.preprocess(df))["test"]
preds = predictor.predict(test_df).iloc[:10]
print(preds)