from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch
from transformers import BitsAndBytesConfig
from llama_index import VectorStoreIndex, StorageContext, Document, ServiceContext, PromptTemplate
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM
from llama_index.vector_stores import PineconeVectorStore
from pinecone import Pinecone, PodSpec
from datasets import load_dataset
import os
import random
from typing import Optional
from transformers import AutoTokenizer


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

def messages_to_prompt(messages):
  prompt = ""
  for message in messages:
    if message.role == 'system':
      prompt += f"<|system|>\n{message.content}\n"
    elif message.role == 'user':
      prompt += f"<|user|>\n{message.content}\n"
    elif message.role == 'assistant':
      prompt += f"<|assistant|>\n{message.content}\n"
# ensure we start with a system prompt, insert blank if needed
  if not prompt.startswith("<|system|>\n"):
    prompt = "<|system|>\n\n" + prompt

  # add final assistant prompt
  prompt = prompt + "<|assistant|>\n"

  return prompt

llm = HuggingFaceLLM(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    query_wrapper_prompt=PromptTemplate("<|system|>\n\n<|user|>\n{query_str}\n<|assistant|>\n"),
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={"quantization_config": quantization_config},
    generate_kwargs={"temperature": 0.3, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    device_map="auto",

)

pc = Pinecone(
    api_key = os.getenv('PINECONE_API_KEY')
    )

try:
    pc.create_index(
        name='my-index',
        dimension=1024,
        metric='euclidean',
        spec=PodSpec(
            replicas= 1, 
            shards= 1, 
            pod_type="p1",
            environment='gcp-starter'
        )        
    )
    print('Index created')
except:
    print('Index already exists')

pinecone_index = pc.Index("my-index")

vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
    add_sparse_vector=True,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:WhereIsAI/UAE-Large-V1", chunk_size=2048)

index = VectorStoreIndex.from_documents(
    [], storage_context=storage_context, service_context=service_context
)

query_engine = index.as_query_engine(streaming=True)


tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
special_tokens = tokenizer.special_tokens_map

app = FastAPI()


async def generate_response(message):
    streaming_response = query_engine.query(
        message,
    )
    for text in streaming_response.response_gen:
        for special_token in special_tokens.values():
            if special_token in text:
                text = text.replace(special_token, "")
        yield text


@app.get("/stream")
async def stream(message: Optional[str] = None):
    if message: 
        return StreamingResponse(generate_response(message))
    else:
        return "No message recieved"
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)