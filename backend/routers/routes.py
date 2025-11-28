from fastapi import APIRouter, HTTPException

from llm.llm import AnalysisLLM, ClassifierLLM, GeneratingLLM
from llm.models import AnalysisResponseFormat, ClassifierResponseFormat, GeneratingResponseFormat

allm = AnalysisLLM()
cllm = ClassifierLLM()
gllm = GeneratingLLM()

api_router = APIRouter()


@api_router.get("/analysis")
async def analysis(mail: str):
    return AnalysisResponseFormat.model_validate_json(allm.invoke_message(message=mail))


@api_router.get("/classifier")
async def classifier(mail: str):
    return ClassifierResponseFormat.model_validate_json(cllm.invoke_message(message=mail))


@api_router.get("/generate")
async def generate(mail: str, mail_class: str):
    if mail_class not in [
        'Запрос информации/документов',
        'Официальная жалоба или претензия',
        'Регуляторный запрос',
        'Партнёрское предложение',
        'Запрос на согласование',
        'Уведомление или информирование'
    ]:
        raise HTTPException(404, 'mail class does not exist')
    return GeneratingResponseFormat.model_validate_json(gllm.invoke_message(message=mail, mail_class=mail_class))
