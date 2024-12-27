# Data-Variability and its importance

   1) Implementation
   The experiment involved the following steps:

   1.1 Importing Necessary Libraries
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from lightgbm import LGBMClassifier, LGBMRegressor
   import warnings
   warnings.filterwarnings('ignore')
   ```
   1.2 Read dataset and inspect dataset 
   ```python
   df = pd.read_csv('sample_data/diabetes.csv')
   df.head()
   ```

   1.3 Display the dimensions of the Data frame (used to quickly check the size of a dataset)
   ```python
   df.shape
   ```
   1.4 Get Data frame Summary
   ```python
   df.info()
   ```
   provides a concise summary of the Data frame df.  It outputs key information about the structure and content of the  dataset, which is helpful for understanding its characteristics before  further analysis.

   1.5 Get descriptive statistics for the numerical columns in the Data frame 
   ```python
   df.describe()
   ```
   1.6 Check the number of missing (NaN) values in each column of the Data frame
   ```python
   df.isna().sum()
   ```
   It returns the count of missing values for each column, helping to identify which features have incomplete data.

   1.7 Make copy of Data frame
   ```python
   new_df = df.copy()
   ```
   new_df = df.copy() creates a new Data frame called new_df as a copy of the existing Data frame df,changes to new_df do not affect df.

   1.8 Drop Outcome column
   ```python
   new_df.drop( 'Outcome', axis = 1, inplace = True)
   ```
   The outcome column has been removed from the dataset as it does not contribute to the analysis for this experiment.

   1.9 Inspect dataset
   ```python
   new_df.head()
   ```
   ```python
   new_df.shape
   ```
   
   1.10 Get data frame columns
   ```python
   df_columns = new_df.columns
   df_columns
   ```
   Extracts the column names from the new_df data frame. This is useful for inspecting or iterating through the column names in the Data frame.

   1.11 Visualizing original dataset
   ```python
   def display_plot( columns, dataframe, plot_type, x_axis ):
      no_col = 2
      no_row = int(np.ceil(len(df_columns)/no_col))
      fig, axs = plt.subplots( no_row, no_col, figsize = (20, 20))

      for i, col in enumerate(columns):
         axis_arg = 'x' if x_axis else 'y'
         plot_type(data=dataframe, **{axis_arg: col}, ax=axs[i // no_col, i % no_col])
         axs[i//no_col, i%no_col].set_title(f'{col} Distribution')
         axs[i//no_col, i%no_col].set_xlabel(col)
         axs[i//no_col, i%no_col].set_ylabel('Count')

      plt.tight_layout()
      plt.show()
   ```
   The provided code defines a function called display_plot that takes several arguments to create a grid of plots for a DataFrame
   Function Parameters:
   columns: A list of column names from the DataFrame to plot.
   dataframe: The DataFrame containing the data for plotting.
   plot_type: A function reference for the desired plot type (e.g., sns.hist, sns.boxplot).

   Plot a histogram for the columns in the original data frame
   ```python
   display_plot( df_columns, df, sns.histplot, True )
   ```
   This will generate a grid of histograms, where each histogram represents the distribution of a column in the data frame. 

   Plot a box plot for the columns in the original data frame
   ```python
   display_plot( df_columns, df, sns.boxplot, False )
   ```
   This will generate a grid of box plots, where each plot represents the distribution of a column in the Data Frame.

   1.12 Get first quartile of each column
   ```python
   Q1 = new_df.quantile(0.25)
   Q1
   ```
   This line of code calculates the first quartile (Q1) of each column in new_df .
   
   1.13 Get third quartile of each column
   ```python
   Q3 = new_df.quantile(0.75)
   Q3
   ```
   This line of code calculates the third quartile (Q3) of each column in new_df .
   
   1.14 Calculate Interquartile Range (IQR)
   ```python
   IQR = Q3 - Q1
   IQR
   ```
   
   1.15 Get lower bound of each column
   ```python
   lower_bound = Q1 - 1.5 * IQR
   lower_bound
   ```
   
   1.16 Get upper bound of each column
   ```python
   upper_bound = Q3 + 1.5 * IQR
   upper_bound
   ```
   
   1.17 Get number of outliers based on the specified upper and lower bound
   ```python
   outlier_mask = (new_df > upper_bound) | (new_df < lower_bound)
   outlier_count = outlier_mask.any(axis=1).sum()
   outlier_count
   ```
   
   1.18 Perform winsorization
   ```python
   winsorized_df = new_df.clip( lower=lower_bound, upper=upper_bound, axis=1 )
   winsorized_df.head()
   ```
   The clip method restricts the values in the data frame to a specified range
   
   1.19 Display the dimensions of the Data frame 
   ```python
   winsorized_df.shape
   ```
   
   1.20 Get descriptive statistics for the numerical columns in the Data frame
   ```python
   winsorized_df.describe()
   ```
   
   1.21 Create side-by-side visualizations for columns 
   ```python
   def display_before_after_plot( columns, dataframes, plot_type, method, x_axis ):
      for i, col in enumerate( columns ):
         fig, axs = plt.subplots( 1, 2, figsize = (20, 5))
         for j, dataframe in enumerate(dataframes):
            axis_arg = 'x' if x_axis else 'y'
            plot_type(data=dataframe, **{axis_arg: col}, ax=axs[ j ])
            if j == 0:
            axs[j].set_title(f'{col} Before { method }')
            else:
            axs[j].set_title(f'{col} After { method } ')

            axs[j].set_xlabel(col)
            axs[j].set_ylabel('Count')

         plt.tight_layout()
         plt.show()
   ```

   Create a histogram for all columns in the original data frame and the winsorized data frame.
   ```python
   dataframes = [new_df, winsorized_df]
   display_before_after_plot( df_columns, dataframes, sns.histplot, 'Winsorization', True )
   ```
   
   Create a box plot for all columns in the original data frame and the winsorized data frame.
   ```python
   display_before_after_plot( df_columns, dataframes, sns.boxplot, 'Winsorization', False )
   ```
   
   1.22 Filter data frame 
   ```python
   new_df2 = new_df[ ( ( new_df >= lower_bound ) & ( new_df <= upper_bound ) ) ]
   new_df2.head()
   ```
   This code will produce NaN for entries that do not satisfy the condition, leaving gaps in the Data frame.
   
   1.23 Inspect data frame 
   ```python
   new_df2.shape
   ```
   
   1.24 Get descriptive statistics for the numerical columns in the Data frame
   ```python
   new_df2.describe()
   ```
   
   1.25 Check for missing values 
   ```python
   new_df2.isna().sum().sort_values()
   ```
   checks for missing values (NaNs) in the new_df2 Data frame, sums them for each column, and then sorts the results in ascending order.

   1.26 Make copy of data frame 
   ```python
   pred_df = new_df2.copy()
   ```
   pred_df = new_df2.copy() creates a new Data frame called pref_df as a copy of the existing Data frame new_df2, changes to pred_df do not affect new_df2.
   
   1.27 Predict Missing Values
   ```python
   def predict_missing_values(new_df2, columns, pred_df ):
      for col in columns:
         #if more than 50% of the column data is missing, drop column
         if new_df2[col].isna().sum() > new_df2.shape[ 0 ] / 2:
            new_df2.drop(col, axis=1, inplace=True)
         else:
            col_missing_index = np.where( new_df2[col].isna() == True )[0]

            #No missing index
            if len( col_missing_index ) == 0:
               continue
            else:
               #create is_nan column in the dataframe with value 0
               new_df2['is_nan'] = 0

               new_df2.loc[col_missing_index, 'is_nan'] = 1

               train = new_df2[ new_df2['is_nan'] == 0 ]
               test = new_df2[ new_df2['is_nan'] == 1 ]

               X_train = train.drop([col, 'is_nan'], axis=1)
               y_train = train[col]

               X_test = test.drop([col, 'is_nan'], axis=1)
               y_test = test[col]

               lgbm = LGBMRegressor()
               lgbm.fit(X_train, y_train)

               y_pred = lgbm.predict(X_test)

               pred_df.loc[col_missing_index, col] = y_pred
   ```
   The function predict_missing_values  is designed to handle missing values in a dataset by predicting them  using a machine learning model. This approach involves training a model  on non-missing data and using the trained model to predict the missing  values in the dataset.
   Function Parameters:
   pred_df: The dataframe containing the data with missing values.
   columns: List of columns to predict missing values for.
   new_df: The dataframe where the predicted values will be stored.

   ```python
   predict_missing_values(new_df2, df_columns, pred_df )
   ```
   
   1.28 Inspect data frame 
   ```python
   pred_df.head()
   ```
   ```python
   pred_df.shape
   ```
   
   1.29 Get descriptive statistics for the numerical columns in the Data frame
   ```python
   pred_df.describe()
   ```

   Create a histogram for all columns in the original data frame and the prediction method data frame.
   ```python
   dataframes = [new_df, pred_df]
   display_before_after_plot( df_columns, dataframes, sns.histplot, 'Prediction', True )
   ```
   
   Create a boxplot for all columns in the original data frame and the prediction method data frame.
   ```python
   display_before_after_plot( df_columns, dataframes, sns.boxplot, 'Prediction', False )
   ```

   1.30 Visualize all data frames

   ```python
   def display_plot3( columns, dataframes, plot_type, x_axis ):
   for i, col in enumerate( columns ):
      fig, axs = plt.subplots( 1, 3, figsize = (20, 5))
      for j, dataframe in enumerate(dataframes):
         axis_arg = 'x' if x_axis else 'y'
         plot_type(data=dataframe, **{axis_arg: col}, ax=axs[ j ])
         if j == 0:
         axs[j].set_title(f'Origina {col} Feature')
         elif j == 1:
         axs[j].set_title(f'Winsorized {col} Feature')
         else:
         axs[j].set_title(f'Prediction {col} Feature')

         axs[j].set_xlabel(col)
         axs[j].set_ylabel('Count')

      plt.tight_layout()
      plt.show()
   ```
   display_plot3 is designed to create plots for specified columns across multiple data frames.
   Function Parameters 
   It loops through columns (columns) and creates subplots for each column across multiple dataframes.
   The plot_type is dynamically passed to define the type of plot (e.g., histogram, boxplot).
   The axis (x or y) is determined by the x_axis .

   Create a histogram for all columns in the original data frame, the winsorized data frame, and the prediction method data frame.

   ```python
   dataframes = [df, winsorized_df, pred_df]
   display_plot3( df_columns, dataframes, sns.histplot, True )
   ```

   Create a box plot for all columns in the original data frame, the winsorized data frame, and the prediction method data frame.

   ```python
   display_plot3( df_columns, dataframes, sns.boxplot, False )
   ```

To gain a deeper understanding and enhance your reading experience, feel free to explore more in my Medium article by clicking [here](https://medium.com/@adedokunjuliusayobami/data-variability-and-its-importance-36f07cf57870)