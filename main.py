import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,  accuracy_score, auc



def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['Approved', 'Rejected'])
    plt.yticks(tick_marks, ['Approved', 'Rejected'])

    thresh = cm.max() / 2.
    for i, j in ((i, j) for i in range(cm.shape[0]) for j in range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


df = pd.read_csv('loan_approval_dataset.csv')
pd.set_option('display.max_columns', None)

print(df.head())
print(df.isnull().sum())

df_eda = df.copy()

loan_status_counts = df_eda[' loan_status'].value_counts()
print(' ')
print(loan_status_counts)
print(' ')
plt.figure(figsize=(6, 4))
plt.bar(loan_status_counts.index, loan_status_counts.values, color=['blue', 'red'])
plt.title('Loan Status Distribution')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.show()

approved_graduated_count = df[(df[' loan_status'] == ' Approved')  & (df[' no_of_dependents'] == 5)][' loan_status'].shape[0]
print("approved aaa", approved_graduated_count)

approved_graduated_count = df[(df[' no_of_dependents'] == 5) ][' no_of_dependents'].shape[0]
print("aaaaa", approved_graduated_count)


sns.violinplot(x=' loan_status', y=' cibil_score', data=df_eda)
df_eda = df.copy()

education_counts = df_eda[' education'].value_counts()
print(' ')
print(education_counts)
print(' ')
plt.figure(figsize=(6, 6))
plt.pie(education_counts, labels=education_counts.index, autopct='%1.1f%%', colors=['red', 'blue'])
plt.title('Education')
plt.show()

df_eda = df.copy()

education_counts = df_eda[' self_employed'].value_counts()
print(' ')
print(education_counts)
print(' ')
plt.figure(figsize=(6, 6))
plt.pie(education_counts, labels=education_counts.index, autopct='%1.1f%%', colors=['red', 'blue'])
plt.title('Employement')
plt.show()

sns.countplot(x = ' no_of_dependents', data = df_eda).set_title('Number of Dependents')




df_eda.columns = df_eda.columns.str.strip()

if ' no_of_dependents' in df_eda.columns and 'loan_status' in df_eda.columns:
    education_loan_status_counts = df_eda.groupby([' no_of_dependents', 'loan_status']).size().unstack()

    # Plot the bar chart
    plt.figure(figsize=(8, 6))
    education_loan_status_counts.plot(kind='bar', stacked=True)
    plt.title('Loan Status by Number of dependents')
    plt.xlabel('Number of dependents')
    plt.ylabel('Count')
    plt.legend(title='Loan Status', loc='upper right', labels=['Approved', 'Rejected'])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
if 'education' in df_eda.columns and 'loan_status' in df_eda.columns:
    education_loan_status_counts = df_eda.groupby(['education', 'loan_status']).size().unstack()

    # Plot the bar chart
    plt.figure(figsize=(8, 6))
    education_loan_status_counts.plot(kind='bar', stacked=True)
    plt.title('Loan Status by Education Level')
    plt.xlabel('Education Level')
    plt.ylabel('Count')
    plt.legend(title='Loan Status', loc='upper right', labels=['Approved', 'Rejected'])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

if 'self_employed' in df_eda.columns and 'loan_status' in df_eda.columns:
    education_loan_status_counts = df_eda.groupby(['self_employed', 'loan_status']).size().unstack()

    # Plot the bar chart
    plt.figure(figsize=(8, 6))
    education_loan_status_counts.plot(kind='bar', stacked=True)
    plt.title('Loan Status by Employement Level')
    plt.xlabel('Employement Level')
    plt.ylabel('Count')
    plt.legend(title='Loan Status', loc='upper right', labels=['Approved', 'Rejected'])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

bin_edges = [300,400,500,600,700,800,900]

# Bin labels
bin_labels = ['301-400', '401-500', '501-600', '601-700', '701-800','801-900']

# Bin the 'cibil_score' column
df_eda['cibil_score_range'] = pd.cut(df_eda['cibil_score'], bins=bin_edges, labels=bin_labels, right=False)

# Check if 'cibil_score_range' and 'loan_status' are in DataFrame columns
if 'cibil_score_range' in df_eda.columns and 'loan_status' in df_eda.columns:
    # Group the data by 'cibil_score_range' and 'loan_status' and count the occurrences
    cibil_loan_status_counts = df_eda.groupby(['cibil_score_range', 'loan_status']).size().unstack()

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    cibil_loan_status_counts.plot(kind='bar', stacked=True)
    plt.title('Loan Status by CIBIL Score Range')
    plt.xlabel('CIBIL Score Range')
    plt.ylabel('Count')
    plt.legend(title='Loan Status', loc='upper right', labels=['Approved', 'Rejected'])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()





# Movable Assets
df_eda['Movable_assets'] = df_eda['bank_asset_value'] + df_eda['luxury_assets_value']

#Immovable Assets
df_eda['Immovable_assets'] = df_eda['residential_assets_value'] + df_eda['commercial_assets_value']

# Drop columns
df_eda.drop(columns=['bank_asset_value','luxury_assets_value', 'residential_assets_value', 'commercial_assets_value'], inplace=True)

fig, ax = plt.subplots(1,2,figsize=(10,5))
sns.histplot(x  = 'Movable_assets', data = df_eda, ax=ax[0], hue = 'loan_status', multiple='stack')
sns.histplot(x =  'Immovable_assets', data = df_eda, ax=ax[1], hue  = 'loan_status', multiple='stack')

df_ml = df.copy()

df_ml.drop(columns=['loan_id'], inplace=True)

# print(df_ml.columns)

# Remove leading spaces from column names
df_ml.rename(columns=lambda x: x.strip(), inplace=True)

# Display the updated DataFrame
print(df_ml.columns)


# Create a LabelEncoder instance
label_encoder = LabelEncoder()

# Apply label encoding to the 'education' column
df_ml['education'] = label_encoder.fit_transform(df_ml['education'])

# Apply label encoding to the 'self_employed' column
df_ml['self_employed'] = label_encoder.fit_transform(df_ml['self_employed'])

# Apply label encoding to the 'loan_status' column
df_ml['loan_status'] = label_encoder.fit_transform(df_ml['loan_status'])

# Display the updated DataFrame with encoded columns
print(df_ml[['education', 'self_employed','loan_status']])


# Create a StandardScaler instance
scaler = StandardScaler()

# Define the feature columns (X) and target column (y)
x = df_ml.drop(columns=['loan_status'])  # Drop 'loan_status' column to get feature columns
y = df_ml['loan_status']  # Target variable

# Select only the numerical columns for scaling (excluding 'loan_status')
numerical_columns = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
                      'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value',
                      'bank_asset_value']

# Apply scaling to the numerical columns
x[numerical_columns] = scaler.fit_transform(x[numerical_columns])

# Display the scaled feature variables (X) and the target variable (y)
print("Scaled Feature Variables (x):")
print(x.head())

print("\nTarget Variable (y):")
print(y.head())


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(x_train.values, y_train.values)

y_pred_dt = decision_tree.predict(x_test)

cm_dt = confusion_matrix(y_test, y_pred_dt)

plot_confusion_matrix(cm_dt, 'Decision Tree Confusion Matrix')

plt.figure(figsize=(20, 10))
plot_tree(decision_tree, filled=True, feature_names=x_train.columns, class_names=[str(i) for i in decision_tree.classes_], rounded=True)
plt.show()



# RandomForestClassifier
random_forest = RandomForestClassifier(random_state=42)

random_forest.fit(x_train.values, y_train.values)
y_pred_rf = random_forest.predict(x_test)

cm_rf = confusion_matrix(y_test, y_pred_rf.round())

plot_confusion_matrix(cm_rf, 'RandomForestClassifier Matrix')


# LogisticRegression
logistic_regression = LogisticRegression(random_state=42)

logistic_regression.fit(x_train.values, y_train.values)
y_pred_lr = logistic_regression.predict(x_test)

cm_lr = confusion_matrix(y_test, y_pred_lr.round())
plot_confusion_matrix(cm_lr, "LogisticRegression Matrix")


# Розрахунок ймовірностей для тестової вибірки
y_prob_lr = logistic_regression.predict_proba(x_test)[:, 1]

# Розрахунок ROC-кривої та AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob_lr)


# Побудова ROC-кривої
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC)')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Logistic Regression')
plt.legend(loc="lower right")
plt.show()

# NaiveBayes
naive_bayes = GaussianNB()

naive_bayes.fit(x_train.values, y_train.values)
y_pred_nb = naive_bayes.predict(x_test)

cm_nb=confusion_matrix(y_test,y_pred_nb.round())
plot_confusion_matrix(cm_nb,"NaiveBayes Matrix")

y_prob = naive_bayes.predict_proba(x_test)[:, 1]  # Probabilities of the positive class
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Naive Bayes Classifier')
plt.legend(loc="lower right")
plt.show()

# Create a KNeighborsClassifier instance
knn = KNeighborsClassifier()

knn.fit(x_train.values, y_train.values)
y_pred_knn = knn.predict(x_test)

cm_knn = confusion_matrix(y_test, y_pred_knn.round())
plot_confusion_matrix(cm_knn,"KNN Matrix")

y_prob = knn.predict_proba(x_test)[:, 1]  # Probabilities of the positive class
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for KNN Classifier')
plt.legend(loc="lower right")
plt.show()


from sklearn.metrics import confusion_matrix, precision_recall_curve, auc

# Assuming cm_dt, cm_rf, cm_lr, cm_svm, cm_knn are confusion matrices for each model
models = ['Decision Tree', 'Random Forest', 'Log Regression', 'Naive Bayes', 'KNN']
false_predictions = [
    (cm_dt[0, 1] + cm_dt[1, 0]),
    (cm_rf[0, 1] + cm_rf[1, 0]),
    (cm_lr[0, 1] + cm_lr[1, 0]),
    (cm_nb[0, 1] + cm_nb[1, 0]),
    (cm_knn[0, 1] + cm_knn[1, 0])
]

# Створення DataFrame
models_df = pd.DataFrame({'Model': models, 'False predictions': false_predictions})

# Сортування DataFrame
models_df = models_df.sort_values(by='False predictions', ascending=False)

# Визначення кольорів для кожної моделі
colors = plt.cm.tab10(np.arange(len(models)))


# Побудова Precision-Recall Curve для кожної моделі
plt.figure(figsize=(10, 6))
for model, name in zip([decision_tree, random_forest, logistic_regression, naive_bayes, knn], models):
    y_prob = model.predict_proba(x_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'{name}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for ML Models')
plt.legend(loc='lower left')
plt.show()

# Define colors for each bar
colors = plt.cm.tab10(np.arange(len(models)))

# Plot
plt.figure(figsize=(10, 6))
bars = plt.barh(models_df['Model'], models_df['False predictions'], color=colors)
plt.xlabel('False Predictions')
plt.ylabel('Model')
plt.title('Comparing ML Algorithms')
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, str(int(bar.get_width())),
             va='center', ha='left')
plt.show()


# Accuracies of all 5 algorhytmes
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Accuracy of Decision Tree:", accuracy_dt)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Accuracy of Random Forest:", accuracy_rf)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Accuracy of Logistic Regression:", accuracy_lr)

accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Accurace of Naive Bayes:", accuracy_nb)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("Accuracy of K-Nearest Neighbors:", accuracy_knn)

