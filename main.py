from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import GoogleSerperAPIWrapper

load_dotenv()

# PDF'i yükle
pdf_loader = PyPDFLoader("tip.pdf")
docs = pdf_loader.load()

# Böl
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# Vektör veritabanı burada weaviate kullanılabilir
embedding_s = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorestore = FAISS.from_documents(chunks, embedding_s)
retriever = vectorestore.as_retriever()

# LLM
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.1)

# Web araması
web_search = GoogleSerperAPIWrapper()

# PDF'ten arama için zincir
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the following context to answer the question:\n\n{context}"),
    ("human", "{input}")
])

qa_chain_pdf = create_stuff_documents_chain(llm, prompt_template)
search_pdf_chain = create_retrieval_chain(retriever, qa_chain_pdf)


def search_pdf(query: str) -> str:
    result = search_pdf_chain.invoke({"input": query})
    return result["answer"] if "answer" in result else str(result)

def summarize_pdf(query:str) -> str:
    summarize_chain = load_summarize_chain(llm, chain_type="stuff")
    return summarize_chain.invoke({"input_documents": chunks})["output_text"]

def search_web(query: str) -> str:
    return web_search.run(query)


system_prompt = (
    "You are a highly capable AI assistant specialized in answering questions using a PDF document named 'tip.pdf'. "
    "Always try to find the answer using the PDF first.\n\n"
    "If the PDF does not contain the answer, or the information is incomplete, you are allowed to use a web search tool to supplement or find missing information.\n\n"
    "Your steps:\n"
    "1. Prioritize the 'pdf_document_search' tool. Use it to extract relevant information from the PDF.\n"
    "2. Only if the PDF lacks the required information or the question relates to current events or general knowledge outside the PDF, then use 'web_search'.\n"
    "3. Always clearly state which source you used (e.g., 'According to the PDF...' or 'Based on web results...').\n"
    "4. Do not guess. If no reliable source is found, politely state that the information is not available.\n"
"Give all your answers in clear, short and understandable TURKISH sentences."
)


# Araçlar tanımla
tools = [
    Tool(
        name="summarize_pdf",
        func=summarize_pdf,
        description="Use this tool to summarize the entire PDF document in Turkish."
    ),
    Tool(
        name="pdf_document_search",
        func=search_pdf,
        description=(
            "Use this tool as your **first and primary method** to answer user questions. "
            "It searches the PDF file 'tip.pdf' for relevant information. "
            "Only use this tool to answer questions directly related to the PDF content (e.g., medical info, facts from the document)."
            "Give all your answers in clear, short and understandable TURKISH sentences."

        )
    ),
    Tool(
        name="web_search",
        func=search_web,
        description=(
            "Use this tool **only when** the answer cannot be found in the PDF or when the user asks about:\n"
            "- Current events\n"
            "- Weather, news, or live data\n"
            "- Topics not mentioned in the PDF (e.g., unrelated general knowledge)\n"
            "Clearly cite the web as the source when you use this tool."
        )
    )
]

# Agent başlat
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs={"system_message": system_prompt}
)

# Sorgu yap
query = " bu pdf ne hakkında? "
response = agent_executor.invoke({"input": query})
print(response["output"])