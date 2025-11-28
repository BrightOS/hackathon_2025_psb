import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time

CLASSES_DESCRIPTION = {
    'complaint': 'Официальная жалоба или претензия',
    'document_request': 'Запрос информации/документов',
    'partnership': 'Партнёрское предложение',
    'regulatory': 'Регуляторный запрос',
    'approval_request': 'Запрос на согласование',
    'notification': 'Уведомление или информирование'
}

print("ОБУЧЕНИЕ МОДЕЛИ КЛАССИФИКАЦИИ ПИСЕМ")
print("=" * 50)

df = pd.read_csv('emails.csv')
print(f"Загружено записей: {len(df)}")
print("Распределение по классам:")
print(df['label'].value_counts())

texts = df['text'].values
labels = df['label'].values

vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    stop_words=['и', 'в', 'на', 'с', 'по', 'о', 'от']
)
X = vectorizer.fit_transform(texts)
print(f"Размерность данных: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"Обучающая выборка: {X_train.shape[0]} писем")
print(f"Тестовая выборка: {X_test.shape[0]} писем")

print("СРАВНЕНИЕ АЛГОРИТМОВ")

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42, probability=True),
    'Naive Bayes': MultinomialNB()
}

results = {}

for name, model in models.items():
    print(f"Обучение {name}...")
    start_time = time.time()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    training_time = time.time() - start_time
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'time': training_time
    }

    print(f"  {name}: Точность = {accuracy:.2%}, Время = {training_time:.2f}с")

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_accuracy = results[best_model_name]['accuracy']

print(f"Лучшая модель: {best_model_name}")
print(f"Точность: {best_accuracy:.2%}")

sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

print("Рейтинг моделей:")
for i, (name, result) in enumerate(sorted_results, 1):
    print(f"  {i}. {name}: {result['accuracy']:.2%} (время: {result['time']:.2f}с)")

y_pred_best = best_model.predict(X_test)

print("Отчет по классификации:")
print(classification_report(y_test, y_pred_best))

print("Примеры предсказаний:")
for i in range(min(5, len(X_test))):
    actual = y_test[i]
    predicted = y_pred_best[i]
    status = "CORRECT" if actual == predicted else "WRONG"
    print(f"{status} Правильный: {actual:15} -> Предсказание: {predicted:15}")

joblib.dump(best_model, 'email_classifier.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

model_info = {
    'best_model_name': best_model_name,
    'best_accuracy': best_accuracy,
    'all_results': results,
    'classes_description': CLASSES_DESCRIPTION,
    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
}
joblib.dump(model_info, 'model_info.pkl')

print(f"Сохранена модель: {best_model_name}")
print(f"Точность сохраненной модели: {best_accuracy:.2%}")


def predict_email_type(text):
    classifier_loaded = joblib.load('email_classifier.pkl')
    vectorizer_loaded = joblib.load('vectorizer.pkl')
    text_vector = vectorizer_loaded.transform([text])
    prediction = classifier_loaded.predict(text_vector)[0]
    probabilities = classifier_loaded.predict_proba(text_vector)[0]

    confidence = max(probabilities)
    class_indices = np.argsort(probabilities)[-3:][::-1]
    top_predictions = [
        (classifier_loaded.classes_[idx], probabilities[idx])
        for idx in class_indices
    ]

    return prediction, confidence, top_predictions


test_texts = [
    "Жалоба на банкомат который не выдает деньги",
    "Прошу справку о доходах для кредита",
    "Предлагаем сотрудничество по страхованию",
    "В соответствии с положением ЦБ просим предоставить отчетность",
    "Уведомляем об изменении реквизитов компании"
]

print("Тестирование на новых текстах:")
for text in test_texts:
    pred, conf, top3 = predict_email_type(text)
    print(f"Текст: '{text}'")
    print(f"  Результат: {pred} (уверенность: {conf:.2%})")
    print(f"  Топ-3 предсказания:")
    for class_name, prob in top3:
        print(f"    - {class_name}: {prob:.2%}")

print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
print(f"Лучшая модель: {best_model_name} с точностью {best_accuracy:.2%}")