# rag.py (inside the deployment folder)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

def create_qa_chain():
    # Paths to your deployment artifacts
    model_path = "rag/model1"
    tokenizer_path = "rag/tokenizer1"
    chroma_db_path = "rag/chroma_db1"

    # Load the model on CPU with full weight loading
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    # Set model configuration for generation
    model.config.max_length = 500
    model.config.max_new_tokens = 100

    # Load the tokenizer (use use_fast=False if needed)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

    # Create a text-generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id
    )

    # Wrap the pipeline as a LangChain LLM with generation parameters
    llm = HuggingFacePipeline(
        pipeline=generator,
        model_kwargs={"max_new_tokens": 100, "max_length": 500}
    )

    # Load the Chroma vector store from the deployment folder
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
    vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embedding)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Define the prompt template for RetrievalQA
    prompt_template = """Use the following passage to answer the question.

Passage: {context}

Question: {question}
Answer:"""
    QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Build and return the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    return qa_chain

def get_answer(query, qa_chain=None):
    """
    Given a query and an optional prebuilt qa_chain, return the answer.
    If the qa_chain is None or raises a meta tensor error, it will be reinitialized.
    """
    if qa_chain is None:
        qa_chain = create_qa_chain()
    try:
        answer = qa_chain.run(query=query)
    except NotImplementedError:
        # Reinitialize the chain if a meta tensor error occurs
        qa_chain = create_qa_chain()
        answer = qa_chain.run(query=query)
    return answer, qa_chain
