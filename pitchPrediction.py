import pandas as pd
from sklearn.linear_model import LogisticRegression


training_data = pd.read_csv('training.csv').dropna()


X_training = training_data[['Velo', 'SpinRate', 'HorzBreak', 'InducedVertBreak']]
y_training = training_data['InPlay']


logistic_model = LogisticRegression()
logistic_model.fit(X_training, y_training)


deploy_data = pd.read_csv('deploy.csv').dropna()


deploy_predictions_proba = logistic_model.predict_proba(deploy_data)[:, 1]  


deploy_data['Predicted_Probability_InPlay'] = deploy_predictions_proba

output_file_path_consolidated = 'deploy_with_predictions_consolidated.csv'
deploy_data.to_csv(output_file_path_consolidated, index=False)


deploy_data.head(), output_file_path_consolidated
