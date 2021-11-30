# F1-Data-Analysis

### (1) Selected Visualizations

![corr_heatmap_vf](https://user-images.githubusercontent.com/91103273/144107992-5cb62ac8-a679-44ea-9506-3dddd67d6e3e.jpeg)

![hamilton_finish_gap](https://user-images.githubusercontent.com/91103273/144107898-c3c4a76f-08c2-4f8a-aa4d-5ea9c9f6888c.jpeg)

![finishing_status_vf](https://user-images.githubusercontent.com/91103273/144107918-610f2c48-1d7c-4218-aa37-9f8804f91fb4.jpeg)

![legacy_constructor_vf](https://user-images.githubusercontent.com/91103273/144107945-b06381c2-dc5c-483b-a19b-0dd98bb5f91d.jpeg)


### (2) Model Scaling Function

- I decided to use the `pipeline` feature from the `scikit-learn` package to create a sequence of scaling transformations and model-fitting operations on the dataset. Applying a transformer and model estimator separately (i.e. not using pipeline) will result in fitted training features being wrongly included in the test-fold of `GridSearchCV`.

- According to the `pipeline` documentation, "pipelines help avoid leaking statistics from your test data into the trained model in cross-validation, by ensuring that the same samples are used to train the transformers and predictors."

- In laymen’s terms, if you separate feature scaling and model-fitting functions while using `GridSearchCV`, you will be creating a biased testing dataset that already contains information about the training set — not good.

```
prediction_scorecard = {'model':[],
                        'accuracy_score':[],
                        'precision_score':[],
                        'recall_score':[],
                        'best_params':[]}

def prediction_model(model_type, model_id):
    # Scale numeric features using 'StandardScaler' and 'One-Hot Encode' categorical features
    scoring = ['neg_log_loss', 'accuracy']
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('ohe', OneHotEncoder(handle_unknown = 'ignore'))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                   ('cat', categorical_transformer, categorical_features)])
    pipeline = Pipeline(steps=[('prep', preprocessor), 
                               (model_id, model_type)])
    return pipeline
```

### (3) Model Results

- I decided to standardize how the model was classified and presented using the following code:

```
def model_results(X_test, model, model_id):
    # Predict!
    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)
    df_pred = pd.DataFrame(np.around(pred_proba, 4), index=X_test.index, columns=['prob_0', 'prob_1'])
    df_pred['prediction'] = list(pred)
    df_pred['actual'] = y_test['podium']
    df_pred['grid_position'] = X_test['grid_position']

    # Include row if an 'actual' or 'predicted' podium occured for calculating accuracy
    df_pred['sort'] = df_pred['prediction'] + df_pred['actual']
    df_pred = df_pred[df_pred['sort'] > 0]
    df_pred.reset_index(inplace=True)
    df_pred = df_pred.groupby(['round']).apply(pd.DataFrame.sort_values, 'prob_1', ascending=False)
    df_pred.drop(['sort'], axis=1, inplace=True)
    df_pred.reset_index(drop=True, inplace=True) 
    
    # Save Accuracy, Precision, 
    prediction_scorecard['model'].append(model_id)
    prediction_scorecard['accuracy_score'].append(accuracy_score(df_pred['actual'], df_pred['prediction']))
    prediction_scorecard['precision_score'].append(precision_score(df_pred['actual'], df_pred['prediction']))
    prediction_scorecard['recall_score'].append(recall_score(df_pred['actual'], df_pred['prediction']))
    prediction_scorecard['best_params'].append(str(model.best_params_))
    display(df_pred.head(10))
    
    
# Support Vector Machines

svm_params= {'svm__C': [0.1, 0.01, 0.001],
             'svm__kernel': ['linear', 'poly', 'rbf'],
             'svm__degree': [1, 2, 3],
             'svm__gamma': [0.1, 0.01, 0.001]}

svm_cv = GridSearchCV(prediction_model(SVC(probability=True), 'svm'),
                      param_grid=svm_params,
                      scoring=scoring, 
                      refit='neg_log_loss',  
                      verbose=10)

# Train Model
svm_cv.fit(X_train, y_train)

# Test Model
model_results(X_test, svm_cv, 'Support Vector Machines')
display(pd.DataFrame(prediction_scorecard))
```

### 2.a. Support Vector Machines Example

<img width="542" alt="Screen Shot 2021-11-30 at 1 36 40 PM" src="https://user-images.githubusercontent.com/91103273/144107332-9338361f-06bf-4a18-b94a-8d01b9e9591c.png">

### 2.b. ML Summary Output

<img width="520" alt="Screen Shot 2021-11-30 at 1 39 03 PM" src="https://user-images.githubusercontent.com/91103273/144107675-e447e16d-09a7-4c46-92d3-9a9bbc895712.png">

