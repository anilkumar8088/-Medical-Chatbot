from flask import Flask, render_template, jsonify, request
# ⚠️ WARNING: Update 'src.helper' if it still uses deprecated HuggingFaceEmbeddings
from src.helper import download_hugging_face_embeddings 
from langchain_pinecone import PineconeVectorStore
# ✅ FIX: Changed 'langchain-google-genai' to 'langchain_google_genai' 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import * # Assuming system_prompt is defined here
import os

# --- Flask App Initialization ---
app = Flask(__name__)
load_dotenv()

# --- Environment Setup ---
# LangChain components automatically pick up these standard environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

# NOTE: The custom os.environ lines for non-standard keys are removed as they are unnecessary 
# and potentially confusing for LangChain. Ensure your .env file is correct.

# --- Embedding Model Initialization ---
# Assuming download_hugging_face_embeddings() returns an Embeddings object
embeddings = download_hugging_face_embeddings() 

# --- Pinecone Vector Store Setup ---
# ✅ FIX: Changed index name to be all lowercase as suggested by the ValueError
index_name = "medical-chatbot-index" 

try:
    # Connect to the existing Pinecone index.
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

except ValueError as e:
    print(f"Error connecting to Pinecone: {e}")
    # Handle the error gracefully, e.g., exit or use a placeholder
    retriever = None # or some dummy retriever

# --- LangChain RAG Chain Setup ---
if retriever:
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)
    
    # Define the chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create the RAG chain components
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
else:
    # Fallback or error state for the RAG chain
    def dummy_invoke(query):
        return {"answer": "Error: RAG system failed to initialize. Check Pinecone configuration."}
    rag_chain = dummy_invoke


# --- Flask Routes ---

@app.route("/")
def index():
    """Renders the main chat HTML page."""
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    """Handles the user's chat message and returns the LLM response."""
    # We use request.form to get data from a form submission (POST)
    msg = request.form.get("msg")
    
    if not msg:
        return "No message provided", 400

    print("User Query:", msg)

    # Invoke the RAG chain
    response = rag_chain.invoke({"input": msg})
    
    answer = response["answer"]
    print("Response : ", answer)
    
    # Return the answer as a string
    return str(answer)


# --- Application Run ---
if __name__ == '__main__':
    # Flask app will run on http://0.0.0.0:8080
    app.run(host="0.0.0.0", port=8080, debug=True)