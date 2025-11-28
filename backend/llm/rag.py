from langchain_community.document_loaders import (
    DirectoryLoader
)
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class DocumentProcessor:
    def __init__(self):
        self.data_directory = './llm/docs'
        self.persist_directory = "./llm/chroma_db"
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    def load_documents(self):
        loaders = [
            DirectoryLoader(self.data_directory, glob="**/*.txt"),
        ]

        all_documents = []
        for loader in loaders:
            try:
                documents = loader.load()
                for doc in documents:
                    doc.metadata["loaded_at"] = "2024-01-01"
                    doc.metadata["document_type"] = "regulatory"
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error loading documents: {e}")
                
        return all_documents

    def split_documents(self, documents):
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        return text_splitter.split_documents(documents)

    def create_vector_store(self, chunks):
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        # vector_store.persist()
        return vector_store

    def process_all_documents(self):
        print("Загрузка документов...")
        documents = self.load_documents()
        print(f"Загружено {len(documents)} документов")

        print("Разделение на чанки...")
        chunks = self.split_documents(documents)
        print(f"Создано {len(chunks)} чанков")

        print("Создание векторной базы...")
        vector_store = self.create_vector_store(chunks)
        print("Векторная база создана и сохранена")

        return vector_store
