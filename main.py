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


#name of the application to run
app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf])

#'server btwn api and client to upload a pdf the inngest function will forward it to the API'



