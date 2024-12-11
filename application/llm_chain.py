import os
from dotenv import load_dotenv
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain.llms import OpenAI
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)
load_dotenv(".env")

openai_api_base = os.getenv("OPENAI_API_BASE")
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model_id = os.getenv("OPENAI_MODEL_ID")

llm = OpenAI(
    openai_api_key = openai_api_key,
    api_base = openai_api_base,
    model_name = openai_model_id,
    temperature = 0.1
)

with open("./data/prompts/extract_features.txt", "r") as f:
    # Create a LangChain PromptTemplate
    extract_features_prompt = ChatPromptTemplate([
        ("system", "You are a helpful assistant"),
        ("user",  f.read())
])

parser = JsonOutputParser()
extract_query_features_chain = extract_features_prompt | llm | parser