import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from customer_types import RAGQueryResult, RAGSearchResult, RAGUpsertResult, RAGChuckAndSrc


#load the env variables that are inside this .env file'
load_dotenv()

#'create clients'
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()

)

#'create an inngest function since this is AI heavy'
#'server btwn API and client e.g, using the front end to upload the pdf files. the inngest server will forward it to the API and will go thru the process of logging it and tracing errors'

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf") #event that can can trigger from a client or from antoher function; whenever this
)

#we want the orchestration logic on top of the API
#Create an inngest function; server between our api and the inngest server
#the inngest server will forwward it to the API;
#setting up the function

async def rag_ingest_pdf(ctx: inngest.Context):
#  return{"hello": "world"}
    def _load(ctx: inngest.Context) -> RAGChuckAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChuckAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChuckAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChuckAndSrc)
    ingested = await ctx.step.run("embed-and-upstart", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump() #convert to json or python format

#another inngest function for querying the PDF files that we ingested and stored in the Vector DB
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int=5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k",3)) #retrieve the 3 most relevant chunnks

    found = await ctx.step.run("RAGResponse-embed-and-search", lambda :_search(question,top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the retrieved document excerpts below to answer the question.\n\n"
        "=== RETRIEVED CONTEXT ===\n"
        f"{context_block}\n"
        "=== END CONTEXT ===\n\n"
        f"Question: {question}\n\n"
        "Provide a clear, concise answer based only on the retrieved context."
    )

    #using OPENAI with my key.
    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini" #using a mini LLM so it isn't too expensive
      # model="gpt-5.6-luna" #current - 8/29/26 - generation option; cost sensative
    )

    #generate the response from the question asked
    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            #"max_completion_tokens": 1024, #with "gpt-5.6-luna"
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer the user's question using only the provided context. "
                        "Do not use outside knowledge. "
                        "If the context does not contain enough information to answer "
                        "the question, say that the information was not found in the document."
                    )
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        }
    )
    #default from openai. we can pass multiple things
    answer = res["choices"][0]["message"]["content"].strip() #strip out leading or trailing white spaces
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}





#name of the application to run
app = FastAPI()

#this is where you add the functions to be made available in inngest
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])

#'server btwn api and client to upload a pdf the inngest function will forward it to the API'



