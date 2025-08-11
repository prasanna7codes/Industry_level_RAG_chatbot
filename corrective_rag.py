import os
from pprint import pprint
from typing import List

from dotenv import load_dotenv
load_dotenv()

from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.document_loaders import FireCrawlLoader
#from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

# Make sure these imports are at the top of your file
from langchain_community.document_loaders import FireCrawlLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores.utils import filter_complex_metadata

from firecrawl import FirecrawlApp
from langchain.docstore.document import Document

from langchain_pinecone import Pinecone






# --- 2. DOCUMENT LOADING AND PROCESSING ---

urls = [
    "https://www.thworks.org/",
]

# Use a list to store the loaded documents
docs_list = []

print("--- Scraping URLs using the compatible FireCrawlLoader ---")
for url in urls:
    try:
        # This loader will now work because you installed firecrawl-py==0.0.20
        loader = FireCrawlLoader(
            api_key=os.environ["FIRECRAWL_API_KEY"],
            url=url,
            mode="scrape"  # "scrape" mode for single URLs
        )
        # Load the document and add it to our list
        docs = loader.load()
        docs_list.extend(docs)
        print(f"--- Successfully scraped: {url} ---")

    except Exception as e:
        print(f"--- FAILED to scrape {url}. Error: {e} ---")
        print("--- Please verify your API key is correct and active on the Firecrawl website. ---")


# Check if any documents were successfully scraped before proceeding
if not docs_list:
    print("\n--- ERROR: No documents were scraped. Exiting because there is no content to process. ---")
    exit() # Stop the script


filtered_docs = filter_complex_metadata(docs_list)

# Split the combined documents into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=100
)
doc_splits = text_splitter.split_documents(docs_list)

print(f"\n--- Successfully loaded and split {len(docs_list)} documents into {len(doc_splits)} chunks. ---")

# --- 3. VECTOR STORE AND RETRIEVER ---

# Use Google's embeddings model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- 3. VECTOR STORE AND PINEcone Multi-Tenant RETRIEVER ---

# This is the name of the index you created in the Pinecone dashboard
index_name = "corrective-rag-app" 

# For this example, we'll pretend we're processing data for "client-abc-123"
# In a real app, this would be dynamic based on which user is logged in.
client_id_namespace = "client-abc-123" 


print(f"\n--- Adding documents to Pinecone index '{index_name}' in namespace '{client_id_namespace}' ---")

# The from_documents method will add the documents to the specified namespace.
# If the index does not exist, it will create it.
vectorstore = Pinecone.from_documents(
    documents=doc_splits,
    embedding=embedding_model,
    index_name=index_name,
    namespace=client_id_namespace  # This is the key to isolating client data
)

print(f"--- Creating retriever scoped to namespace '{client_id_namespace}' ---")
# The retriever created from this vectorstore will automatically search only
# within the 'client-abc-123' namespace.
retriever = vectorstore.as_retriever()

print("--- Pinecone vector store and retriever are ready. ---")




# --- 4. SETUP LLMS AND GRADERS ---

# Initialize Gemini models
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
llm_grader = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, model_kwargs={"response_mime_type": "application/json"})

# --- Retrieval Grader ---
prompt_retrieval_grader = PromptTemplate(
    template="""You are a grader assessing the relevance of a retrieved document to a user's question.
    If the document contains keywords or semantic meaning related to the user question, grade it as relevant.
    Your goal is to filter out erroneous retrievals.
    Give a binary score 'yes' or 'no'. Provide the output in JSON format with a single key 'score'.
    \n\n
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question}""",
    input_variables=["question", "document"],
)
retrieval_grader = prompt_retrieval_grader | llm_grader | JsonOutputParser()

# --- Hallucination Grader ---
prompt_hallucination_grader = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in and supported by a set of facts.
    Give a binary score 'yes' or 'no' to indicate whether the answer is supported by the facts.
    Provide the output in JSON format with a single key 'score'.
    \n\n
    Here are the facts: \n\n {documents} \n\n
    Here is the answer: {generation}""",
    input_variables=["generation", "documents"],
)
hallucination_grader = prompt_hallucination_grader | llm_grader | JsonOutputParser()

# --- Answer Grader ---
prompt_answer_grader = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful in resolving a question.
    Give a binary score 'yes' or 'no' to indicate its usefulness.
    Provide the output in JSON format with a single key 'score'.
    \n\n
    Here is the answer: \n\n {generation} \n\n
    Here is the question: {question}""",
    input_variables=["generation", "question"],
)
answer_grader = prompt_answer_grader | llm_grader | JsonOutputParser()

# --- RAG Generation Chain ---
prompt_rag = PromptTemplate(
    template="""You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Keep the answer concise and use a maximum of three sentences.
    \n\n
    Question: {question}
    Context: {context}
    Answer:""",
    input_variables=["question", "context"],
)
rag_chain = prompt_rag | llm | StrOutputParser()

# --- Web Search Tool ---
web_search_tool = TavilySearch(k=3)


# --- 5. LANGGRAPH IMPLEMENTATION ---

# Define the state for the graph
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question.
        generation: The LLM's generated answer.
        web_search: A flag indicating whether a web search is needed.
        documents: A list of retrieved documents.
    """
    question: str
    generation: str
    web_search: str
    documents: List[Document]

# --- Graph Nodes ---
def retrieve(state):
    print("---NODE: RETRIEVE DOCUMENTS---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state):
    """
    Grades documents based on relevance.
    After grading, it decides whether to trigger a web search based on
    whether ANY relevant documents were found.
    """
    print("---NODE: GRADE DOCUMENTS (Alternative Logic)---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.get('score', 'no')
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    
    # --- The New Logic is Here ---
    # Only trigger a web search if the list of relevant documents is empty
    if not filtered_docs:
        # If all documents are irrelevant, trigger a web search
        print("---DECISION: No relevant documents found. Web search is required.---")
        web_search = "yes"
    else:
        # If at least one document is relevant, proceed to generation
        print(f"---DECISION: Found {len(filtered_docs)} relevant documents. Proceeding to generation.---")
        web_search = "no"

    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def generate(state):
    print("---NODE: GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]
    
    # Format documents for the RAG chain
    context = "\n\n".join(doc.page_content for doc in documents)
    
    generation = rag_chain.invoke({"context": context, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def web_search(state):
    print("---NODE: WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])
    
    # Perform web search
    search_results = web_search_tool.invoke(question) # No need for a dictionary here either
    
    # --- THIS IS THE FIX ---
    # The new TavilySearch returns a list of strings, not a list of dicts.
    # So we can join them directly.
    web_results = "\n\n".join(search_results) 
    web_results_doc = Document(page_content=web_results)
    
    documents.append(web_results_doc)
    
    return {"documents": documents, "question": question}

# --- Conditional Edges ---
def decide_to_generate(state):
    print("---EDGE: DECIDE TO GENERATE---")
    web_search = state["web_search"]
    
    if web_search == "yes":
        print("---DECISION: WEB SEARCH REQUIRED---")
        return "websearch"
    else:
        print("---DECISION: PROCEED TO GENERATION---")
        return "generate"

def grade_generation_and_decide_next_step(state):
    print("---EDGE: GRADE GENERATION---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Check for hallucinations
    hallucination_score = hallucination_grader.invoke({"documents": [doc.page_content for doc in documents], "generation": generation})
    hallucination_grade = hallucination_score.get("score", "no")
    
    if hallucination_grade.lower() == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCS---")
        # Check if the answer is useful
        answer_score = answer_grader.invoke({"question": question, "generation": generation})
        answer_grade = answer_score.get("score", "no")
        if answer_grade.lower() == "yes":
            print("---DECISION: GENERATION IS USEFUL, FINISH---")
            return "useful"
        else:
            print("---DECISION: GENERATION IS NOT USEFUL, RETRY WITH WEB SEARCH---")
            return "not useful"
    else:
        print("---DECISION: GENERATION HAS HALLUCINATIONS, RETRY---")
        return "not supported"

# --- Build the Graph ---
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("websearch", web_search)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"websearch": "websearch", "generate": "generate"},
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_and_decide_next_step,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)

# Compile the graph
app = workflow.compile()

# Test the graph
inputs = {"question":"what tech stack do you use ?"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"--- Step: {key} ---")
        pprint(value, indent=2, width=120)
    print("\n" + "="*80 + "\n")