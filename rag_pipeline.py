from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Replicate

query = "lightweight transformers for language tasks"
arxiv_docs = ArxivLoader(query=query, load_max_docs=3).load()

pdf_data = []

for doc in arxiv_docs:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    texts = text_splitter.create_documents([doc.page_content])
    pdf_data.append(texts)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
db = Chroma.from_documents(pdf_data[0], embedding=embeddings)

llm = Replicate(
    model="meta/meta-llama-3-70b-instruct",
    model_kwargs={"temperature": 0.0, "top_p": 1, "max_new_tokens": 1000},
)

qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=db.as_retriever()
)

question = "Overview of language models, their applications, and their capabilities?"
answer = qa({"query": question})
answer
