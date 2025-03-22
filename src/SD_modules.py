import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(data):

  #Handling 'Date' column
  data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(DAY=1))
  data.set_index('Date', inplace=True)
  data.drop(columns=['Year', 'Month'], inplace=True)
  data = data.reset_index()

  temp_data = data.copy()

  #Separate the features
  numeric_columns = temp_data.drop(columns=['Crop', 'Region', 'Date']).columns.tolist()
  categorical_columns = ['Crop', 'Region']
  data_to_transform = temp_data.drop(columns=['Date'])  # Exclude 'Date' column

  # Use ColumnTransformer to apply transformations to specific columns
  preprocessor = ColumnTransformer(
      transformers=[
          ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_columns),  # One-Hot Encoding for categorical variables
          ('num', StandardScaler(), numeric_columns)  # StandardScaler for numeric variables
      ],
      remainder='drop'  # This keeps the remaining columns (like Year and Month) intact
  )

  # Apply the transformations to the dataset
  processed_data = preprocessor.fit_transform(data)

  # Get the transformed column names for categorical columns (after one-hot encoding)
  encoded_columns = preprocessor.transformers_[0][1].get_feature_names_out(categorical_columns)

  # Combine the new column names with the original columns
  columns = list(encoded_columns) + numeric_columns

  temp_data = pd.DataFrame(processed_data, columns=columns)

  # Now you have the transformed data with Year and Month intact
  temp_data['Date'] = data['Date']
  data = temp_data

  return data

def preprocess_dtypes(data):
  data = data.select_dtypes(include=['number'])
  return data
