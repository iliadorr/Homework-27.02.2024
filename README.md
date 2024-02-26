# Homework-27.02.2024
Homework for Mr Tauheed

import pandas as pd
import numpy as np
from scipy.stats import zscore

HA: Data_cleaning
Handling Missing Data Questions
Question 1: How do you identify and handle missing values in a Pandas DataFrame?

Identifying and handling missing values is a crucial task in data analysis using Pandas. Missing values can occur due to various reasons like data collection errors, data corruption, or simply because the information is not available.

A missing value in a dataset is displayed as a question mark, zero, NaN or just a blank cell. But how can we deal with missing data?

Of course, every situation is different and should be evaluated differently.

There are many ways to deal with missing values. Let's look at typical options using the example of a dataset - 'Titanic'. This data is an open Kaggle dataset.

Identification missing values
There are two methods for detecting missing data: - isnull() and notnull().

The Pandas read.csv() method is used for loading. The file path is quoted in brackets, so that Pandas reads the file into the Dataframes from that address. The file path can be a URL address or your local file address.


train_df = pd.read_csv('Titanic-Dataset.csv')


The result is a boolean value indicating whether the value passed to the argument is indeed missing. "True" ( True ) means that the value is a missing value, and "False" ( False ) means that the value is not missing.

missing_data = train_df.isnull()
missing_data.head(6)
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
0	False	False	False	False	False	False	False	False	False	False	True	False
1	False	False	False	False	False	False	False	False	False	False	False	False
2	False	False	False	False	False	False	False	False	False	False	True	False
3	False	False	False	False	False	False	False	False	False	False	False	False
4	False	False	False	False	False	False	False	False	False	False	True	False
5	False	False	False	False	False	True	False	False	False	False	True	False
notnull() method in Pandas is used to detect non-missing values in a DataFrame. It returns a boolean DataFrame where True indicates that the value is not missing, and False indicates that the value is missing. As an example, let's create some df DataFrame, check non-null values and output them, also let's find out which cabins in the dataset are non-null and output them. Let's write a code that will find the non-zero values in the 'Cabin' column and output them to us.

data = {'A': [1, 2, np.nan, 4],
        'B': [np.nan, 5, 6, 7],
        'C': ['a', 'b', 'c', 'd']}
df = pd.DataFrame(data)

# Using notnull() to detect non-missing values
not_missing_data = df.notnull()
print(not_missing_data)
       A      B     C
0   True  False  True
1   True   True  True
2  False   True  True
3   True   True  True
not_missing_data = pd.notnull(train_df['Cabin'])
train_df[not_missing_data]
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
1	2	1	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	female	38.0	1	0	PC 17599	71.2833	C85	C
3	4	1	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	female	35.0	1	0	113803	53.1000	C123	S
6	7	0	1	McCarthy, Mr. Timothy J	male	54.0	0	0	17463	51.8625	E46	S
10	11	1	3	Sandstrom, Miss. Marguerite Rut	female	4.0	1	1	PP 9549	16.7000	G6	S
11	12	1	1	Bonnell, Miss. Elizabeth	female	58.0	0	0	113783	26.5500	C103	S
...	...	...	...	...	...	...	...	...	...	...	...	...
871	872	1	1	Beckwith, Mrs. Richard Leonard (Sallie Monypeny)	female	47.0	1	1	11751	52.5542	D35	S
872	873	0	1	Carlsson, Mr. Frans Olof	male	33.0	0	0	695	5.0000	B51 B53 B55	S
879	880	1	1	Potter, Mrs. Thomas Jr (Lily Alexenia Wilson)	female	56.0	0	1	11767	83.1583	C50	C
887	888	1	1	Graham, Miss. Margaret Edith	female	19.0	0	0	112053	30.0000	B42	S
889	890	1	1	Behr, Mr. Karl Howell	male	26.0	0	0	111369	30.0000	C148	C
204 rows × 12 columns

Now let's see how many missing values are in each column using train_df.isnull().sum(). train_df.isnull().sum() calculates the total number of missing values in each column of the train_df. We use .use() function in this situation.

train_df.isnull().sum()
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
Handling missing values
Remove missing values
The most obvious thing we can do is to remove these missing values. We can delete rows and columns with missing values using dropna(). As a first example, let's remove rows with missing values from a DataFrame that was created earlier.

df.dropna(inplace=True)
print(df)
     A    B  C
1  2.0  5.0  b
3  4.0  7.0  d
Now let's delete all rows with passengers who have unknown cabin numbers and output the result.

passengers_with_known_cabins = train_df.dropna(subset='Cabin')
print(passengers_with_known_cabins)
     PassengerId  Survived  Pclass  \
1              2         1       1   
3              4         1       1   
6              7         0       1   
10            11         1       3   
11            12         1       1   
..           ...       ...     ...   
871          872         1       1   
872          873         0       1   
879          880         1       1   
887          888         1       1   
889          890         1       1   

                                                  Name     Sex   Age  SibSp  \
1    Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
3         Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
6                              McCarthy, Mr. Timothy J    male  54.0      0   
10                     Sandstrom, Miss. Marguerite Rut  female   4.0      1   
11                            Bonnell, Miss. Elizabeth  female  58.0      0   
..                                                 ...     ...   ...    ...   
871   Beckwith, Mrs. Richard Leonard (Sallie Monypeny)  female  47.0      1   
872                           Carlsson, Mr. Frans Olof    male  33.0      0   
879      Potter, Mrs. Thomas Jr (Lily Alexenia Wilson)  female  56.0      0   
887                       Graham, Miss. Margaret Edith  female  19.0      0   
889                              Behr, Mr. Karl Howell    male  26.0      0   

     Parch    Ticket     Fare        Cabin Embarked  
1        0  PC 17599  71.2833          C85        C  
3        0    113803  53.1000         C123        S  
6        0     17463  51.8625          E46        S  
10       1   PP 9549  16.7000           G6        S  
11       0    113783  26.5500         C103        S  
..     ...       ...      ...          ...      ...  
871      1     11751  52.5542          D35        S  
872      0       695   5.0000  B51 B53 B55        S  
879      1     11767  83.1583          C50        C  
887      0    112053  30.0000          B42        S  
889      0    111369  30.0000         C148        C  

[204 rows x 12 columns]

Replace missing values
When dealing with missing data in a Pandas DataFrame, one approach is to fill in those missing values. The fillna() method provides a convenient way to achieve this. Let's work with the titanic dataset. For the example, let's replace the age value with the average passengers with missing age value. To do this, lets us find the average value of people's age and replace missing values with it. For convenience, lets take a particular person with a missing age value, for example the person from line 6 - Mr. James.

train_df['Age'].fillna(train_df['Age'].mean(), inplace = True)
print(train_df.iloc[5])
PassengerId                   6
Survived                      0
Pclass                        3
Name           Moran, Mr. James
Sex                        male
Age                   29.699118
SibSp                         0
Parch                         0
Ticket                   330877
Fare                     8.4583
Cabin                       NaN
Embarked                      Q
Name: 5, dtype: object
C:\Users\MZBes\AppData\Local\Temp\ipykernel_16228\3161201996.py:1: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  train_df['Age'].fillna(train_df['Age'].mean(), inplace = True)
Interpolation
To deal with missing data in a dataset, interpolation is a very useful method for estimating and filling the gaps. Interpolation refers to the estimation of missing values using known data points that surround them. This technique works best for time series or ordered data where we expect that the missing values exhibit some pattern.

Pandas provides for interpolate() function which allows for various types of interpolation. Let’s see it with a simple example:

data_interpolation = {'Date': pd.date_range(start='2022-01-01', periods=10),
        'Value': [1, 2, np.nan, 4, np.nan, 6, np.nan, 8, 9, 10]}
df_interpolation = pd.DataFrame(data_interpolation)
print(df_interpolation.interpolate(method='linear'))
        Date  Value
0 2022-01-01    1.0
1 2022-01-02    2.0
2 2022-01-03    3.0
3 2022-01-04    4.0
4 2022-01-05    5.0
5 2022-01-06    6.0
6 2022-01-07    7.0
7 2022-01-08    8.0
8 2022-01-09    9.0
9 2022-01-10   10.0
There, the missing values are replaced with the estimated values obtained through linear interpolation.

Forward Fill (ffill) and Backward Fill (bfill)
Forward fill ffill and backward fill bfill are techniques for filling missing values in a DataFrame by propagating non-null values forward or backward along a specified axis, which is applicable to time series data or sequential data where missing values happen in sequence.

Forward fill ffill
With forward fill method, the missing values get replaced with those of the previous available ones on that particular column, as it pursues the last observed value until the next null value is met.
This way is good when the most recent valid information should be carried through subsequent periods with no information.
Backward fill bfill
The missing values are replaced with those of the succeeding populated cells within that specific axis through utilizing backward fill technique. It moves backwards to propagate till it gets to another previous unavailable value.
In cases where there is an expectation of future valuable observation, back-fill can come into play here since all empty spaces should take up valid measures from their respective future positions.
Let’s see how they work using example:

data_fbfill = {'A': [1, np.nan, 3, np.nan, 5],
        'B': [np.nan, 2, np.nan, 4, np.nan]}
df_fbfill = pd.DataFrame(data_fbfill)

print(df_fbfill.ffill())
print(df_fbfill.bfill())
     A    B
0  1.0  NaN
1  1.0  2.0
2  3.0  2.0
3  3.0  4.0
4  5.0  4.0
     A    B
0  1.0  2.0
1  3.0  2.0
2  3.0  4.0
3  5.0  4.0
4  5.0  NaN

Question 2: What is imputation, and why might it be useful in dealing with missing data?
Missing values in a dataset can be filled using imputation, which is done by estimating or calling up for some calculated value to fill in the missing gaps. Imputing is an important step during data preprocessing since it guarantees that datasets are complete and ready for analysis and modeling. In order to handle missing at random and at the same time maintain the structure of the data, several imputation techniques have been developed.

There are various typical methods of imputation such as:

Mean/Median Imputation:
The mean or median of the column( feature) with missing values will substitute them.
For numerical features this approach is straightforward and often used. As long as they do not bias on the distribution of the data, it assumes missings are lost randomly.
Mode Imputation:
Replacing missing categorical values with mode (most frequently occurring value) for each attribute.
This type of imputation is appropriate for categorical features having missing values.
K-Nearest Neighbors (KNN) Imputation:
Values get imputed by estimating from similar instances within a dataset (nearest neighbors).
KNN imputes takes into account distances between instances and their attributes when determining what values should be inserted.
Regression Imputation:
Use regression models trained on the non-missing values of the dataset to predict those missing.
This technique can be used for both numerical and categorical features, it exploits the relationships among features to impute missing values.
Multiple imputation:
Generate multiple databases of plausible values for each missing value.
In several statistical analyses, multiple imputation is carried out in order to account for the uncertainty stemming from imputed data.
For instance, mean imputation can be done using Pandas as shown below:

data = {'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, 7, 8],
        'C': [np.nan, 10, 11, 12]}
df = pd.DataFrame(data)

df_imputed = df.fillna(df.mean())
df_moded = df.fillna(df.mode())

print('Original DataFrame')
print(df)
print('DataFrame after Mode Imputation: \n')
print(df_moded)
print("\nDataFrame after Mean Imputation:")
print(df_imputed)
Original DataFrame
     A    B     C
0  1.0  5.0   NaN
1  2.0  NaN  10.0
2  NaN  7.0  11.0
3  4.0  8.0  12.0
DataFrame after Mode Imputation: 

     A    B     C
0  1.0  5.0  10.0
1  2.0  7.0  10.0
2  4.0  7.0  11.0
3  4.0  8.0  12.0

DataFrame after Mean Imputation:
          A         B     C
0  1.000000  5.000000  11.0
1  2.000000  6.666667  10.0
2  2.333333  7.000000  11.0
3  4.000000  8.000000  12.0
In this example, missing values in numerical columns 'A' and 'B' are filled with the mean of each respective column using mean imputation. Why imputation might be useful in dealing with data? Here is some points why imputation should be used to handle missing data:

Maintenance of Data Integrity: Imputation ensure that datasets are complete and can be analyzed and modeled comprehensively. The completion of missing values through imputation helps prevent losing valuable data that may take place when incomplete information is discarded.
Preservation of Sample Size: Eliminating rows or columns with missing values may result in considerable reduction in the size of the entire dataset, thereby affecting statistical power for analytical tasks or models. This procedure allows a researcher to keep all elements of sample hence guaranteeing robust and dependable outcomes.
Reduction of Bias: Removal of cases having missing data could lead to bias especially if it was related to the outcome variable. These methods try to compute the values lost by examining the available information prior to running other analysis so as minimize them.
Increase in Statistical Power: Through imputing, more information is obtained which can enhance precision and accuracy making estimations that are statistically powerful. By filling in these gaps, imputations optimizes use of existing information that results in accurate outputs.
Compatible with Analytical Techniques: There are many statistical and machine learning algorithms which require complete datasets for training purposes and testing

Data Transformation Questions
Question 1: How can you encode categorical variables in a Pandas DataFrame?
Encoding categorical variables involves converting categorical data (i.e., data that represents categories or labels) into a numerical format that machine learning algorithms can understand. Categorical variables can take on a limited, fixed number of possible values, such as colors, types of fruit, or levels of education. There are methods of encoding:

Ordinal encoding:
Ordinal encoding is similar to label encoding but allows for custom mapping of categories to specific integers based on a predefined order.
Unlike label encoding, where the mapping is based on the order of appearance, ordinal encoding allows for explicit control over the mapping.
Ordinal encoding is suitable for categorical variables with a natural order or hierarchy, where the numerical labels reflect the inherent order of the categories.
Example: If the original categorical column represents levels of education ('low', 'medium', 'high'), ordinal encoding can map them to integers 1, 2, and 3, respectively, based on their order of importance or level.
data_ordinal = {'category': ['low', 'medium', 'high', 'medium', 'low']}
df_ord = pd.DataFrame(data_ordinal)

# Define the mapping of categories to integers
mapping = {'low': 1, 'medium': 2, 'high': 3}

df_ord['encoded_category'] = df_ord['category'].map(mapping)

print("Original DataFrame:")
print(df_ord)
Original DataFrame:
  category  encoded_category
0      low                 1
1   medium                 2
2     high                 3
3   medium                 2
4      low                 1
Pandas factorize() The pandas factorize() function is used to encode categorical data. It returns a tuple containing two elements:
An array of integers representing the numerical labels of the categorical data.
An Index object that contains the unique categories in the data, along with their corresponding integer codes. Example: the original categorical column 'category' with values ['A', 'B', 'C', 'A', 'C'] is encoded using the factorize() function. The encoded labels are assigned to the labels array, while the unique categories are stored in the unique_categories Index array. The encoded labels are then added as a new column 'encoded_category' to the DataFrame df_fact. Finally, both the original DataFrame and the encoded labels along with their corresponding unique categories are printing for reference.
data_fact = {'category': ['A', 'B', 'C', 'A', 'C']}
df_fact = pd.DataFrame(data_fact)

# Apply to factorize to the categorical column
labels, unique_categories = pd.factorize(df_fact['category'])

# Add the encoded labels to the DataFrame
df_fact['encoded_category'] = labels

print("Original DataFrame:")
print(df_fact)
print("\nEncoded labels:")
print(labels)
print("\nUnique categories:")
print(unique_categories)
Original DataFrame:
  category  encoded_category
0        A                 0
1        B                 1
2        C                 2
3        A                 0
4        C                 2

Encoded labels:
[0 1 2 0 2]

Unique categories:
Index(['A', 'B', 'C'], dtype='object')
Question 2: What is one-hot encoding, and when would you use it in data preprocessing?
One-hot encoding is a technique used to represent categorical variables as binary vectors. In one-hot encoding, each category is represented by a binary feature column, where only one column is active (i.e., contains a value of 1) for each category, and all other columns are inactive (i.e., contain a value of 0).

One-hot encoding transforms categorical variables into a binary format, where each category becomes a separate binary feature column.
For each unique category in the original column, a new binary feature column is created, indicating whether that category is present or not.
One-hot encoding is suitable for categorical variables without an inherent order, as it doesn't impose any numerical relationship between categories.
Example: If the original categorical column has categories 'A', 'B', and 'C', one-hot encoding creates three new binary columns: 'A', 'B', and 'C', where each row has a 1 in the corresponding column if it belongs to that category and 0 otherwise.
data_1h_encoding = {'category': ['A', 'B', 'C', 'A', 'C']}
df_1h = pd.DataFrame(data_1h_encoding)

encoded_1h_df = pd.get_dummies(df_1h, columns=['category'])

print("Original DataFrame:")
print(df_1h)
print("\nDataFrame after One-Hot Encoding:")
print(encoded_1h_df)
Original DataFrame:
  category
0        A
1        B
2        C
3        A
4        C

DataFrame after One-Hot Encoding:
   category_A  category_B  category_C
0        True       False       False
1       False        True       False
2       False       False        True
3        True       False       False
4       False       False        True
One-hot encoding is commonly used in data preprocessing when dealing with categorical variables in machine learning tasks. Here are some scenarios where one-hot encoding is useful:

Categorical Variables with No Inherent Order: One hot encoding is useful, for dealing with variables that lack any order or structure. For example variables such, as "color" (green blue) or "country" (USA, UK, Canada) where the categories don't follow a sequence. In cases one hot encoding proves to be a choice.
Preventing Ordinal Bias: When we use one encoding it helps avoid the model from assuming a hierarchy or order, among categories. For example if we encode a variable using integer labels (known as label encoding) the model could wrongly perceive the encoded values as having an order, which could result in biased predictions.
Preserving Category Information: Using one encoding maintains the information without establishing any numerical relationships, between categories. This guarantees that the model can effectively differentiate between categories during both training and inference stages.
Suitability Machine Learning Algorithms: Machine learning algorithms, including regression, decision trees and support vector machines necessitate numerical input features. With one encoding categorical variables are converted into a format that aligns with these algorithms making them usable in modeling workflows.
Preventing Misinterpretation of Categorical Variables: Through the representation of variables as vectors one hot encoding prevents any misinterpretation of categorical data as continuous or ordinal. This ensures that each category is treated as distinct and independent, from others by the model.
Removing Duplicates Questions:
Question 1: How do you identify and remove duplicate rows from a DataFrame?
Duplicate rows in a DataFrame are rows that have identical values across all columns.

To identify duplicate rows in a DataFrame in Pandas, we can use the duplicated() method. This method returns a boolean Series indicating whether each row is a duplicate of a previous row. For example:

data_dupl = {'A': [1, 2, 3, 4, 2],
        'B': ['a', 'b', 'c', 'd', 'b']}
df_dupl = pd.DataFrame(data_dupl)

# Identify duplicate rows
duplicate_mask = df_dupl.duplicated()

print("Original DataFrame:\n")
print(df_dupl)
print("Duplicate Mask:")
print(duplicate_mask)
Original DataFrame:

   A  B
0  1  a
1  2  b
2  3  c
3  4  d
4  2  b
Duplicate Mask:
0    False
1    False
2    False
3    False
4     True
dtype: bool
To remove duplicate rows from a DataFrame in Pandas, we can use the drop_duplicates() method. This method removes rows that are duplicates of other rows, keeping only the first occurrence of each unique row by default. Here's how we can do it:

data_remove_dupl = {'A': [1, 2, 3, 4, 2],
        'B': ['a', 'b', 'c', 'd', 'b']}
df_remove_dupl = pd.DataFrame(data_remove_dupl)

# Remove duplicate rows
cleaned_df = df_remove_dupl.drop_duplicates()

print("Original DataFrame:\n")
print(df_remove_dupl)
print("DataFrame after removing duplicates:") # Last fourth row has been removed from Original DataFrame
print(cleaned_df)
Original DataFrame:

   A  B
0  1  a
1  2  b
2  3  c
3  4  d
4  2  b
DataFrame after removing duplicates:
   A  B
0  1  a
1  2  b
2  3  c
3  4  d
Question 2: Can you explain the difference between the duplicated() and drop_duplicates() methods in Pandas?
duplicated() and drop_duplicates() are used in Pandas for handling duplicated rows in a DataFrame, but they serve for different purposes:

duplicated():

The duplicated() method is used to identify duplicate rows in a DataFrame.
It returns a boolean Series indicating whether each row is a duplicate of a previous row.
Each duplicate row is marked with a True value, while non-duplicate rows are marked with False.
This method helps in identifying which rows are duplicates without actually modifying the DataFrame.
drop_duplicates():

The drop_duplicates() method is used to remove duplicate rows from a DataFrame.
It returns a new DataFrame with duplicate rows removed, keeping only the first occurrence of each unique row by default.
This method is used to clean the DataFrame by eliminating duplicate entries.
By specifying additional parameters such as subset or keep, you can customize how duplicate rows are identified and which duplicates are kept.
In short, the duplicated() function is utilized to detect duplicate rows, whereas drop_duplicates() is employed to eliminate duplicate rows from the DataFrame. These two functions work hand in hand and are commonly applied in data preprocessing workflows to manage duplicate data efficiently.
Data Scaling and Normalization Questions:

Question 1: Discuss the importance of feature scaling in machine learning.
Machine learning depends on feature scaling as a basic aspect of its ways to ensure that models perform effectively and efficiently. Think of it this way, in some dataset you may be using other features whose measurement units differ or are totally different from each other and thus need to be converted.

Let us explain it further. Consider a dataset which has two attributes namely “age” measured in years and “income” measured in thousands of dollars. It could be something like age ranging from 20 to 80 while income ranges from 
200,000. When data is fed into machine learning algorithms, they might perceive income to be more important than it actually is because the numbers are bigger causing the model’s decision-making process become imbalanced.

This problem can be corrected through feature scaling which ensures that features are transformed such that they exist within the same scale normally between 0 and 1 or have a mean of zero with standard deviation equal to one. Such normalization ensures that all features irrespective of their initial scales or units contribute equally towards model predictions.

Moreover, many machine learning algorithms, converge more efficiently when the features are on a similar scale. This means that feature scaling can significantly speed up the training process and improve the overall performance of our models.

In essence, feature scaling is like standardizing the playing field for our machine learning algorithms, allowing them to interpret and utilize the data more accurately and effectively. It's a critical step that helps us unlock the full potential of our models and produce more reliable and meaningful insights from our data.

Question 2: Explain the difference between min-max scaling and z-score normalization.
Min-max Scaling:

Min-max scaling is a feature scaling technique used to rescale the features to a fixed range, typically between 0 and 1.
It works by subtracting the minimum value of each feature from the original value and then dividing by the difference between the maximum and minimum values.
Min-max scaling ensures that all feature values lie within the specified range, making it useful for algorithms that require input features to have a similar scale.
Formula: min-max-formula
Z-score Normalization:

Z-score normalization, also known as standardization, is a feature scaling technique used to transform the features to have a mean of 0 and a standard deviation of 1.
It works by subtracting the mean of each feature from the original value and then dividing by the standard deviation.
Z-score normalization centers the data around the mean and scales it based on the variability (standard deviation) of the data.
It results in feature values that are centered around 0 and have a similar scale, making it suitable for algorithms that assume a Gaussian distribution of the features.
Formula: z-score-formula
Handling Outliers Questions:
Question 1: What are outliers, and why might they impact machine learning models?
Outliers are distinct data points that deviate significantly from the majority of data in a dataset. They are rare or unusual compared to most data. Outliers can emerge due to measurement errors, experimental variation, or genuine but rare events. Outliers can affect machine learning models in several ways:

Model fitting Bias: The estimates of statistical parameters such as the mean and variance may be mistaken by outliers. In case that there is an outlier in the data, its greatly deviant value will heavily affect the calculation of summary statistics and produce bias for model estimates.

Distortion of Relations: Outliers can distort relationships within a database between variables. Outliers in regression analysis can influence an estimated regression line too much, which in turn results in inaccurate predictions and wrong conclusions about relationships between dependence independent variables.

Increased Variance: Outliers can increase the variance of the model predictions, making the model less reliable. Machine learning models trained on datasets containing outliers may have higher prediction errors and lower generalization performance when applied to new, unseen data.

Sensitive to Noise: Some machine learning algorithms are sensitive to outliers and noise in the data. Outliers can introduce noise into the training process, making it more challenging for the algorithm to identify meaningful patterns and relationships in the data.

When Outliers Cause Model Instability: Model instability can be caused by outliers, particularly in methods that depend on distance or optimization. For example, in clustering algorithms like k-means, outliers can prevent per se clustering and lead to suboptimal assignments of cluster membership.

Overfitting: One of the most insidious consequences of outliers is overfitting. In this scenario, the model becomes overly complex and starts fitting the noise present in the data rather than the true underlying patterns. Accommodating outliers can lead to poor generalization, rendering the model ineffective for new data.

Outliers can also influence feature engineering decisions. Scaling methods, such as Min-Max scaling, can be sensitive to outliers, while more robust methods, like Z-score normalization, are less affected by their presence.

Data distribution: The presence of outliers can distort the underlying data distribution. In datasets with features normally distributed, outliers can make the distribution appear skewed or non-normal. Consequently, models may draw faulty conclusions about the data, impacting their predictions.

In summary, outliers can have a significant impact on machine learning models by biasing parameter estimates, distorting relationships between variables, increasing variance, introducing noise, causing model instability, and leading to overfitting. It is essential to identify and appropriately handle outliers during the data preprocessing stage to ensure the robustness and reliability of machine learning models.

Question 2: Describe different methods for detecting outliers in a dataset in Python
Outliers can impact machine learning models by skewing statistical measures and influencing the model's behavior, leading to inaccurate predictions or biased results. Detecting and handling outliers is crucial for ensuring the robustness and reliability of machine learning models. So, there are few methods to detect them:

Z-score can be used to detect outliers. We determine a threshold value for the Z-score, and data points with Z-scores beyond this threshold are considered potential outliers. It is common to use -3 and +3 as threshold. Any data point with a z-score below -3 or above +3 are treated as outliers. The reason for this common threshold is that 99.7% of the values in a standard normal distribution fall between -3 and +3. z-score-method The suitability of Z-score as a method for detecting outliers depends on the specific characteristics of your data:
Z-score assumes that the data follow a normal distribution. If your data is not normally distributed, Z-score may not be the most appropriate method for detecting outliers.
Z-score is sensitive to extreme values, which can influence the mean and standard deviation, leading to misleading outlier detection.
Choosing a threshold for identifying outliers is subjective. A threshold of 3 is commonly used, but the optimal threshold may depend on your dataset and the context. z-score-method-2
For example, we create the DataFrame with one outlier 100, and we need to find that outlier. We will SciPy library to calculate this:

data_with_outlier = pd.DataFrame({'A': [1, 2, 3, 4, 5, 100]})

# Calculate Z-scores for column 'A'
z_scores = zscore(data_with_outlier['A'])

# Define threshold
threshold = 3

# Find outliers
outliers = data_with_outlier[abs(z_scores) > threshold]

print(outliers)
Empty DataFrame
Columns: [A]
Index: []

