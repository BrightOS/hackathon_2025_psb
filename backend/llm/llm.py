from config import settings
from llm.models import QwenLLM


class BaseLLM:
    SYSTEM_PROMPT = ''

    def __init__(self):
        self.llm = QwenLLM(
            folder_id = settings.MODEL_FOLDER_ID,
            api_key = settings.MODEL_API_KEY
        )

    def invoke_message(self, message: str) -> str:
        return self.llm._call(
            prompt=message,
            system_prompt=self.SYSTEM_PROMPT
        )

class ClassifierLLM(BaseLLM):
    SYSTEM_PROMPT = """
Ты - ИИ-ассистент по обработки корреспонденции для крупного банка.
В твои задачи входит классификация писем на 6 категорий:

- Запрос информации/документов
- Официальная жалоба или претензия
- Регуляторный запрос
- Партнёрское предложение
- Запрос на согласование
- Уведомление или информирование

Формат ответа: JSON с полями:
- "mail_class": строка, строго такая же как и в названиях классов
"""


class AnalysisLLM(BaseLLM):
    SYSTEM_PROMPT = """
Ты - ИИ-ассистент по обработки корреспонденции для крупного банка.
В твои задачи входит извлечение информации из письма, подготовка краткой выжимки и извлечение контактов.

Формат ответа: JSON с полями:
- "summary": строка
- "contacts": строка
"""


class GeneratingLLM(BaseLLM):
    SYSTEM_PROMPT = """
Ты - ИИ-ассистент по обработки корреспонденции для крупного банка.
В твои задачи входит составление ответа на полученное письмо.
Важно соблюсти нужный стиль в зависимости от того, к какому классу относится письмо.
Класс письма - {mail_class}.

Формат ответа: JSON с полями:
- "mail": строка
"""

    def invoke_message(self, message: str, mail_class: str) -> str:
        return self.llm._call(
            prompt=message,
            system_prompt=self.SYSTEM_PROMPT.format(mail_class=mail_class)
        )
