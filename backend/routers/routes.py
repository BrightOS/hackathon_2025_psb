from fastapi import APIRouter, HTTPException

from llm.llm import AnalysisLLM, ClassifierLLM, GeneratingLLM, DocLLM
from llm.rag import DocumentProcessor
from llm.models import AnalysisResponseFormat, ClassifierResponseFormat, GeneratingResponseFormat, DocumentResponseFormat
from kafka import KafkaProducer, KafkaConsumer

processor = DocumentProcessor().process_all_documents()

allm = AnalysisLLM()
cllm = ClassifierLLM()
gllm = GeneratingLLM()
dllm = DocLLM(processor)

api_router = APIRouter()


@api_router.get("/analysis")
async def analysis(mail: str):
    return AnalysisResponseFormat.model_validate_json(allm.invoke_message(message=mail))


@api_router.get("/classifier")
async def classifier(mail: str):
    return ClassifierResponseFormat.model_validate_json(cllm.invoke_message(message=mail))


@api_router.get("/documents")
async def documents(mail: str):
    response = dllm.invoke_message(message=mail)
    # return response
    result = []
    for doc in response['context']:
        curr_res = {}
        curr_res['document_type'] = doc.metadata['document_type']
        curr_res['source'] = doc.metadata['source']
        curr_res['content'] = doc.page_content
        result.append(curr_res)
    return {
        'docs': result
    }

@api_router.get("/send")
async def generate(mail: str, id: str):
    key_bytes = bytes(id, encoding='utf-8') 
    value_bytes = bytes(mail, encoding='utf-8') 

    producer_tosend = KafkaProducer(
        bootstrap_servers='0.0.0.0:9092')
    
    producer_tosend.send('some_topic', key=key_bytes, value=value_bytes) 
    producer_tosend.flush() 

    return "sended"

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
