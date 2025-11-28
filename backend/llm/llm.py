from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from config import settings
from llm.models import QwenLLM


class BaseLLM:
    SYSTEM_PROMPT = ''

    def __init__(self):
        self.llm = QwenLLM(
            folder_id = settings.MODEL_FOLDER_ID,
            api_key = settings.MODEL_API_KEY
        )

    def invoke_message(self, message: str, prev: str) -> str:
        return self.llm._call(
            prompt=message,
            system_prompt=self.SYSTEM_PROMPT.format(prev=prev)
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

Обязательно учти предыдущие сообщения от данного пользователя.
Для контекста переписки данные сообщения предоставляются:
{prev}

Формат ответа: JSON с полями:
- "mail_class": строка, строго такая же как и в названиях классов

Следование формату ответа СТРОГО ОБЯЗАТЕЛЬНО
"""


class AnalysisLLM(BaseLLM):
    SYSTEM_PROMPT = """
Ты - ИИ-ассистент по обработки корреспонденции для крупного банка.
В твои задачи входит извлечение информации из письма, подготовка краткой выжимки и извлечение контактов.

Обязательно учти предыдущие сообщения от данного пользователя.
Для контекста переписки данные сообщения предоставляются:
{prev}

Формат ответа: JSON с полями:
- "summary": строка
- "contacts": строка

Следование формату ответа СТРОГО ОБЯЗАТЕЛЬНО
"""


class GeneratingLLM(BaseLLM):
    SYSTEM_PROMPT = """
Ты - ИИ-ассистент по обработки корреспонденции для крупного банка.
В твои задачи входит составление ответа на полученное письмо.
Важно соблюсти нужный стиль в зависимости от того, к какому классу относится письмо.
Класс письма - {mail_class}. Учти, что письма разных классов нужно писать в разном стиле.

Обязательно учти предыдущие сообщения от данного пользователя.
Для контекста переписки данные сообщения предоставляются:
{prev}

Формат ответа: JSON с полями:
- "mail": строка

Следование формату ответа СТРОГО ОБЯЗАТЕЛЬНО
"""

    def invoke_message(self, message: str, mail_class: str, prev: str) -> str:
        return self.llm._call(
            prompt=message,
            system_prompt=self.SYSTEM_PROMPT.format(mail_class=mail_class, prev=prev)
        )


class DocLLM:
    def __init__(self, vectordb):
        self.llm = QwenLLM(
            folder_id = settings.MODEL_FOLDER_ID,
            api_key = settings.MODEL_API_KEY
        )
        self.system_prompt = """
Используй предоставленный контекст, чтобы ответить на вопрос:
{context}

Формат ответа: JSON с полями:
- "docs": строка

Следование формату ответа СТРОГО ОБЯЗАТЕЛЬНО
"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
        ])

        self.document_chain = create_stuff_documents_chain(self.llm, self.prompt)

        self.retriever = vectordb.as_retriever()
        self.retrieval_chain = create_retrieval_chain(self.retriever, self.document_chain)


    def invoke_message(self, message: str):
        return self.retrieval_chain.invoke({"input": message})
