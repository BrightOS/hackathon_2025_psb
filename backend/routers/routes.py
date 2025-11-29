from fastapi import APIRouter, HTTPException
from kafka import KafkaProducer, KafkaConsumer

from llm.llm import AnalysisLLM, ClassifierLLM, GeneratingLLM, DocLLM
from llm.rag import DocumentProcessor
from llm.models import AnalysisResponseFormat, ClassifierResponseFormat, GeneratingResponseFormat, DocumentResponseFormat, HelperResponseFormat

from db.mongo import MongoManager
from db.model import DBEmailModel

from prometheus_client import Counter, Histogram, generate_latest, REGISTRY, Gauge
from fastapi.responses import Response

LIKE_REQUESTS = Gauge('only_likes_req', 'Total number of like requests')
TOTAL_REQUESTS = Counter('total_req_sum', 'Total number of like requests')

processor = DocumentProcessor().process_all_documents()

allm = AnalysisLLM()
cllm = ClassifierLLM()
gllm = GeneratingLLM()
dllm = DocLLM(processor)

mongo_manager = MongoManager()

api_router = APIRouter()

# prometheus-client==0.19.0
# prometheus-client==0.19.0
@api_router.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(REGISTRY),
        media_type="text/plain"
    )

@api_router.get("/like")
async def like():
    TOTAL_REQUESTS.inc()
    LIKE_REQUESTS.inc()
    return "sended like"

@api_router.get("/dislike")
async def dislike():
    TOTAL_REQUESTS.inc()
    return "sended dis"

@api_router.get("/analysis", response_model = AnalysisResponseFormat)
async def analysis(sendler: str, mail: str):
    message = allm.invoke_message(message=mail, prev=[])
    return AnalysisResponseFormat.model_validate_json(message)


@api_router.get("/classifier", response_model = ClassifierResponseFormat)
async def classifier(sendler: str, mail: str):
    return ClassifierResponseFormat.model_validate_json(cllm.invoke_message(message=mail, prev=[]))


@api_router.get("/documents", response_model = DocumentResponseFormat)
async def documents(sendler: str, mail: str):
    response = dllm.invoke_message(message=mail)
    result = []
    for doc in response['context']:
        curr_res = {}
        curr_res['document_type'] = doc.metadata['document_type']
        curr_res['source'] = doc.metadata['source']
        curr_res['content'] = doc.page_content
        result.append(curr_res)
    docs = {
        'docs': result
    }
    return DocumentResponseFormat.model_validate(docs)

@api_router.get("/send")
async def send(mail: str, id: str):
    key_bytes = bytes(id, encoding='utf-8') 
    value_bytes = bytes(mail, encoding='utf-8') 

    producer_tosend = KafkaProducer(
        bootstrap_servers='0.0.0.0:9092')
    
    producer_tosend.send('some_topic', key=key_bytes, value=value_bytes) 
    producer_tosend.flush() 

    return "sended"

@api_router.get("/generate", response_model = GeneratingResponseFormat)
async def generate(sendler: str, mail: str, mail_class: str):
    if mail_class not in [
        'Запрос информации/документов',
        'Официальная жалоба или претензия',
        'Регуляторный запрос',
        'Партнёрское предложение',
        'Запрос на согласование',
        'Уведомление или информирование'
    ]:
        raise HTTPException(404, 'mail class does not exist')
    return GeneratingResponseFormat.model_validate_json(gllm.invoke_message(message=mail, mail_class=mail_class, prev=[]))


@api_router.get("/helper", response_model = HelperResponseFormat)
async def helper(sendler: str, mail: str):
    cursor = mongo_manager.get_mail(sendler)
    prev = []
    for i in cursor:
        prev.append('-письмо:\n' + i['mail'] + '\n-номер письма: ' + str(i['order']))
    order = len(prev)
    prev_letters = '\n-----------\n'.join(prev)
    mongo_manager.set_mail(
        DBEmailModel(
            sendler=sendler,
            mail=mail,
            order=order
        ).model_dump()
    )

    print(prev_letters)

    analysis_message = AnalysisResponseFormat.model_validate_json(allm.invoke_message(message=mail, prev=prev_letters))
    classifier_message = ClassifierResponseFormat.model_validate_json(cllm.invoke_message(message=mail, prev=prev_letters))
    mail_class = classifier_message.mail_class
    generate_message = GeneratingResponseFormat.model_validate_json(gllm.invoke_message(message=mail, mail_class=mail_class, prev=prev_letters))
    doc_response = dllm.invoke_message(message=mail)
    doc_result = []
    for doc in doc_response['context']:
        curr_res = {}
        curr_res['document_type'] = doc.metadata['document_type']
        curr_res['source'] = doc.metadata['source']
        curr_res['content'] = doc.page_content
        doc_result.append(curr_res)
    docs = {
        'docs': doc_result
    }
    doc_message = DocumentResponseFormat.model_validate(docs)
    return HelperResponseFormat.from_all_formats(
        classifier_message,
        analysis_message,
        generate_message,
        doc_message,
        order
    )
