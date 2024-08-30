import streamlit as st

from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings


embedding = HuggingFaceEmbeddings()
db = FAISS.load_local("faiss", embedding, allow_dangerous_deserialization=True)
    
model = Ollama(model='mistral')

# Prompt
template = """Answer the following question from the pdf files,
{context}
Question: {question}
Helpful Answer:"""
    
QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=db.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True,
    )


st.title("🦜🔗 Streamlit Chatbot")

def generate_response(input_text):
    
    result = qa_chain({"query": input_text})
    
    sources = []
    for s in result["source_documents"]:
        sources.append(",".join([key + ":" + str(s.metadata[key]) for key in s.metadata]))
    sources = sorted(set(sources))
    
    st.info(result["result"])
    for item in sources:
        st.info("Paper source:" + item)


with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "Please type your query",
    )
    
    submitted = st.form_submit_button("Submit")

    generate_response(text)