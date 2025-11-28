import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# Загружаем сохраненные модели
@st.cache_resource
def load_models():
    try:
        classifier = joblib.load('email_classifier.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return classifier, vectorizer
    except FileNotFoundError:
        st.error("Модели не найдены. Сначала обучите модель, запустив train_classifier.py")
        return None, None


classifier, vectorizer = load_models()

# Шаблоны ответов для каждого типа писем
response_templates = {
    'complaint': '''Уважаемый клиент,

Благодарим Вас за обращение в наш банк. Мы внимательно изучим указанную ситуацию и примем все необходимые меры для ее разрешения.

Наши специалисты свяжутся с Вами в течение 2 рабочих дней.

С уважением,
Служба поддержки клиентов''',

    'document_request': '''Уважаемый клиент,

В ответ на Ваш запрос направляем запрашиваемые документы во вложении к настоящему письму.

Если Вам потребуются дополнительные документы или информация, пожалуйста, сообщите нам.

С уважением,
Операционный отдел''',

    'partnership': '''Уважаемые коллеги,

Благодарим за проявленный интерес к сотрудничеству и Ваше предложение.

Наши специалисты рассмотрят его в ближайшее время и свяжутся с Вами для обсуждения деталей.

С уважением,
Отдел развития бизнеса''',

    'general_inquiry': '''Уважаемый клиент,

Благодарим за обращение в наш банк.

По интересующему Вас вопросу предоставляем следующую информацию:
[здесь будет конкретный ответ на вопрос]

Если у Вас остались дополнительные вопросы, мы готовы на них ответить.

С уважением,
Центр клиентского обслуживания''',

    'regulatory': '''Уважаемые коллеги,

В ответ на Ваш запрос сообщаем следующее:
[информация по запросу регулятора]

При необходимости готовы предоставить дополнительную информацию и документы.

С уважением,
Отдел по работе с регуляторами''',

    'approval_request': '''Уважаемые коллеги,

Ваш запрос на согласование получен и передан в соответствующий отдел.

Срок рассмотрения запроса составляет 3 рабочих дня. О результатах согласования будет сообщено дополнительно.

С уважением,
Отдел документооборота''',

    'notification': '''Уважаемый клиент,

Информируем Вас о следующем:
[текст уведомления]

Благодарим за внимание.

С уважением,
Администрация банка'''
}

# Простая маршрутизация - кто должен согласовать
departments_mapping = {
    'complaint': 'Служба поддержки клиентов\nЮридический отдел\nРуководство отделения',
    'document_request': 'Операционный отдел',
    'partnership': 'Отдел развития бизнеса\nЮридический отдел',
    'general_inquiry': 'Центр клиентского обслуживания',
    'regulatory': 'Отдел по работе с регуляторами\nЮридический отдел\nCompliance отдел',
    'approval_request': 'Соответствующий профильный отдел\nЮридический отдел',
    'notification': 'Административный отдел'
}

# Сроки ответа по SLA
sla_mapping = {
    'complaint': '2 рабочих дня (срочно)',
    'document_request': '3 рабочих дня',
    'partnership': '5 рабочих дней',
    'general_inquiry': '1 рабочий день',
    'regulatory': '1 рабочий день (срочно)',
    'approval_request': '3 рабочих дня',
    'notification': '1 рабочий день'
}

# Настройка страницы
st.set_page_config(
    page_title="Банковский помощник",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("Банковский ИИ-помощник")
st.markdown("### Автоматическая обработка входящей корреспонденции")

# Проверка загрузки моделей
if classifier is None or vectorizer is None:
    st.stop()

# Основной интерфейс
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Введите текст письма")
    email_text = st.text_area(
        "Текст входящего письма:",
        height=200,
        placeholder="Вставьте сюда текст письма от клиента, партнера или регулятора...",
        key="email_input"
    )

with col2:
    st.subheader("Инструкция")
    st.info("""
    **Примеры для тестирования:**
    - Жалоба на сотрудника банка
    - Прошу выписку по счету
    - Предлагаем сотрудничество
    - Вопрос по ипотеке
    - Регуляторный запрос ЦБ
    - Уведомление об изменении реквизитов
    """)

    st.subheader("Доступные классы:")
    for class_name, description in {
        'complaint': 'Жалобы',
        'document_request': 'Запросы документов',
        'partnership': 'Партнерские предложения',
        'general_inquiry': 'Общие запросы',
        'regulatory': 'Регуляторные запросы',
        'approval_request': 'Запросы согласования',
        'notification': 'Уведомления'
    }.items():
        st.write(f"• {description}")

# Кнопка анализа
if st.button("Проанализировать письмо", type="primary", key="analyze_btn"):
    if email_text.strip():
        with st.spinner("Анализируем письмо..."):
            try:
                # Преобразуем текст в вектор
                text_vector = vectorizer.transform([email_text])

                # Предсказываем тип письма
                prediction = classifier.predict(text_vector)[0]

                # Получаем уверенность предсказания
                confidence = max(classifier.predict_proba(text_vector)[0])

                # Получаем шаблон ответа
                response_template = response_templates.get(prediction, '')
                departments = departments_mapping.get(prediction, '')
                sla = sla_mapping.get(prediction, '')

                # Отображаем результаты
                st.success("Анализ завершен!")

                # Разделяем результаты на колонки
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Тип письма", prediction)

                with col2:
                    st.metric("Уверенность", f"{confidence:.1%}")

                with col3:
                    st.metric("Срок ответа", sla)

                # Показываем маршрутизацию
                st.subheader("Маршрут согласования:")
                st.write("**Отделы для согласования:**")
                st.text_area(
                    "Список отделов:",
                    value=departments,
                    height=100,
                    key="departments_area"
                )

                st.write("**Срок ответа по SLA:**")
                st.info(sla)

                # Показываем черновик ответа
                st.subheader("Черновик ответа:")
                st.text_area(
                    "Подготовленный ответ:",
                    value=response_template,
                    height=300,
                    key="response_area"
                )

                # Кнопка для копирования
                st.download_button(
                    label="Скачать ответ",
                    data=response_template,
                    file_name=f"ответ_{prediction}.txt",
                    mime="text/plain",
                    key="download_btn"
                )

            except Exception as e:
                st.error(f"Ошибка при анализе письма: {str(e)}")

    else:
        st.error("Пожалуйста, введите текст письма для анализа")

# Боковая панель с информацией
with st.sidebar:
    st.header("О системе")
    st.markdown("""
    **Возможности:**
    - Автоматическая классификация писем
    - Генерация черновиков ответов
    - Определение отделов для согласования
    - Контроль сроков ответа по SLA

    **Технологии:**
    - Machine Learning (scikit-learn)
    - Обработка естественного языка
    - Веб-интерфейс (Streamlit)
    """)

    # Показываем загруженные модели
    st.header("Статус системы")
    st.success("Модели загружены")
    st.info(f"Классификатор: {classifier.__class__.__name__}")
    st.info(f"Количество классов: {len(classifier.classes_)}")

    st.header("Статистика")
    if hasattr(classifier, 'feature_importances_'):
        st.write(f"Количество признаков: {len(classifier.feature_importances_)}")

    st.header("Поддержка")
    st.markdown("""
    При возникновении проблем:
    1. Проверьте что модель обучена
    2. Убедитесь в наличии файлов .pkl
    3. Перезапустите приложение
    """)