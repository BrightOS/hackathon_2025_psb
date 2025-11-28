from langchain_core.language_models import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Optional
from pydantic import BaseModel, Field
import requests

class ClassifierResponseFormat(BaseModel):
    mail_class: str = Field(description="Название класса письма")


class AnalysisResponseFormat(BaseModel):
    summary: str = Field(description="Краткое содержание текста")
    contacts: str = Field(description="Информация о контактах: email, телефон, имя и т.д.")


class GeneratingResponseFormat(BaseModel):
    mail: str = Field(description="Ответное письмо в нужном формате")


class DocumentPartResponseFormat(BaseModel):
    document_type: str = Field(description="Тип документа")
    source: str = Field(description="Файл-источник")
    content: str = Field(description="Содержимое документа")


class DocumentResponseFormat(BaseModel):
    docs: List[DocumentPartResponseFormat] = Field(description="Выжимки из документов")


class HelperResponseFormat(BaseModel):
    mail_class: str = Field(description="Название класса письма")
    summary: str = Field(description="Краткое содержание текста")
    contacts: str = Field(description="Информация о контактах: email, телефон, имя и т.д.")
    new_mail: str = Field(description="Ответное письмо в нужном формате")
    docs: List[DocumentPartResponseFormat] = Field(description="Выжимки из документов")
    count_mails: int


    @classmethod
    def from_all_formats(
        cls,
        a: ClassifierResponseFormat,
        c: AnalysisResponseFormat,
        g: GeneratingResponseFormat,
        d: DocumentResponseFormat,
        count_mails: int
    ):
        return HelperResponseFormat(
            mail_class=a.mail_class,
            summary=c.summary,
            contacts=c.contacts,
            new_mail=g.mail,
            docs=d.docs,
            count_mails=count_mails
        )


class QwenLLM(LLM):
    folder_id: str = ""
    api_key: str = ""

    @property
    def _llm_type(self) -> str:
        return "qwen"

    def _call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        url = "https://llm.api.cloud.yandex.net/v1/chat/completions"
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
            "OpenAI-Project": self.folder_id
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": f"gpt://{self.folder_id}/qwen3-235b-a22b-fp8/latest",
            "messages": messages
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code}, {response.text}")

        result = response.json()
        return result['choices'][0]['message']['content']

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"folder_id": self.folder_id}
