pip install git+https://github.com/tantanchen/dspy.git
pip install groq
pip install colbert
pip install chromadb
pip install langchain
pip install lark
pip install -q streamlit
pip install -U langchain-community

import dspy
import groq
import colbert
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import TextSplitter
import streamlit as st
import pandas as pd
from langchain.vectorstores import Chroma
import numpy as np
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import streamlit as st
class CaseSplitter(TextSplitter):
    def __init__(self):
        super().__init__()

    def split_text(self, file):
        cases = file.strip().split("\n\nRow ")
        return cases


lm = dspy.GROQ(model='mixtral-8x7b-32768', api_key="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",
               max_tokens=1000)
dspy.configure(lm=lm)
df = pd.read_csv("output.csv", encoding="cp1252")


def create_collection(client):
    cases_from_df = np.array_split(df, len(df))
    collection = client.get_or_create_collection("Supreme_court_decisions")
    important_columns = ["decisionType", "dateDecision", "term", "naturalCourt", "caseName", "chief", "dateArgument",
                         "petitioner", "petitionerState", "respondent", "respondentState", "caseOrigin", "caseSource",
                         "issue", "issueArea", "decisionDirection"]
    metadata_columns = [item for item in df.columns.tolist() if item not in important_columns]
    for i in range(len(cases_from_df)):

        res = ""
        metadata = ""
        for column_name in important_columns:
            res += column_name + ": " + str(cases_from_df[i][column_name].item()) + "\n"
        for column_name in metadata_columns:
            metadata += column_name + ": " + str(cases_from_df[i][column_name].item()) + "\n"
        collection.add(
            ids=[str(i)],
            documents=res,
            metadatas=[{"documents": metadata}])


if 'button_clicked' not in st.session_state:
    st.session_state['button_clicked'] = False


class RAG2(dspy.Module):
    def __init__(self):
        super().__init__()
        client = chromadb.Client()
        print(client.list_collections())
        if "Supreme_court_decisions" not in [c.name for c in client.list_collections()]:
            create_collection(client)
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        langchain_chroma = Chroma(
            client=client,
            collection_name="Supreme_court_decisions",
            embedding_function=embedding_function
        )
        self.retrieve = langchain_chroma
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
    def forward(self, question):
        context = self.retrieve.max_marginal_relevance_search(question, k=1)
        answer = self.generate_answer(context=context, question=question)
        print(context)
        return dspy.Prediction(answer=answer.answer)


class RagWithMemory():
    def __init__(self):
        print("RagWithMemory")
        self.rag = RAG2()


    def forward(self, question, history):
        new_prompt_tempalte = f"You are an AI assistant, which gives details about already existing Supreme Court decisions. Consider previous chat history:{history} \nConsider this information in your following answers\n Question: {question}"

        pred = self.rag(new_prompt_tempalte)
        answer = pred.answer

        return answer


def get_llm_response(question, rag):
    answer = rag.forward(question)
    return answer


st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    rag_with_memory = RagWithMemory()

    st.chat_message("user").write(prompt)
    msg = rag_with_memory.forward(prompt, st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
