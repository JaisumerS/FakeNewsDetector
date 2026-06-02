from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def create_pipeline(model_name="1"):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.6, max_features=1000)
    if model_name.lower() == "3":
        classifier = RandomForestClassifier(n_estimators=75, random_state=42, max_depth = 5)
    elif model_name.lower() == "2":
        classifier = MultinomialNB()
    elif model_name.lower() == "1":
        classifier = LinearSVC(random_state=0, C=0.001)
    else:
        raise ValueError("Model not recognized. Choose '1', '2', or '3'.")
    
    return Pipeline([
        ('tfidf', vectorizer),
        ('classifier', classifier)
    ])

def inputters():
    border = "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
    print(border)
    print("Welcome to the Fake News Detection Input Interface")
    print(border)
    title = input("Please enter the news title: ")
    print(border)
    article = input("Please enter the news article text: \n")
    print(border)
    combined = title.strip() + " " + article.strip()
    return combined

def metrics(pipeline, X_train, y_train, X_test, y_test, y_pred):
    param_grid = {
        'tfidf__max_df': [0.6, 0.7],
        'tfidf__max_features': [500, 1000],
        'classifier__C': [0.001, 0.01, 0.1]
    }
    
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print("Best parameters:", grid.best_params_)
    print("Best cross-validation score:", grid.best_score_)
    print("Test set score:", grid.score(X_test, y_test))
    #Now we shall change the parameters so that the svm is more regularized
    
    '''
    Step 4 and 5 of the ML Workflow: Validation and Visualization of Results
    
    Note: The code below is from "Validation Metrics Practice Sol'n" that is in D2L.
    
    '''
    
    # Calculate validation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    precision_recall_auc = average_precision_score(y_test, y_pred)
    
    # Print the metrics
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1-Score: {f1:.6f}")
    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"Precision-Recall AUC: {precision_recall_auc:.6f}")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()
    
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    
    # Plot precision-recall curve
    plt.figure()
    plt.plot(recall, precision, color='red', lw=2, label='Precision-Recall curve (AUC = %0.2f)' % precision_recall_auc)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower right')
    plt.show()
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Set up the figure suize using subplots
    fig, axes = plt.subplots(figsize=(5, 5))
    
    # Create a heatmap
    sns.heatmap(cm, xticklabels=['Real News', 'Fake News'],  yticklabels=['Real News', 'Fake News'],  annot=True, fmt='d', cmap='Blues', ax=axes)
    axes.set_title('Classification Matrix')
    axes.set_xlabel('Predicted')
    axes.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()