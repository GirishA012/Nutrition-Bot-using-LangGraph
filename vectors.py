from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from parser import docs

print(docs)

langchain_docs = []
for doc in docs:
    content = doc.text if hasattr(doc, 'text') else str(doc)
    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
    langchain_docs.append(Document(page_content=content, metadata=metadata))


#Text chunking done
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)


#Splitting documents into chunks
split_docs = text_splitter.split_documents(langchain_docs)

print(f"Split into {len(split_docs)} chunks")

#Created embeddings using open-source model
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}  # Use 'cuda' if you have GPU
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

#Created and saved FAISS vector store
vector_store = FAISS.from_documents(
    documents=split_docs,
    embedding=embeddings
)


vector_store.save_local("nutrition_vector_store")
print("Vector store created and saved locally!")


loaded_vectorstore = FAISS.load_local(
    folder_path="nutrition_vector_store",
    embeddings=embeddings,
    allow_dangerous_deserialization=True  
)