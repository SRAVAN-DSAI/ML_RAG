import json
import torch
import logging
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
import gradio as gr
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load and preprocess JSON datasets
qa_data = []
data_dir = "."  # Adjust if datasets are stored elsewhere
for i in range(1, 15):
    file_name = f"{data_dir}/ml_qa_synthetic_set_{i}.json"
    try:
        with open(file_name, "r") as f:
            data = json.load(f)
            qa_data.extend(data)
            logger.info(f"Loaded {file_name} with {len(data)} Q&A pairs")
    except FileNotFoundError:
        logger.warning(f"{file_name} not found. Skipping...")
        continue

if not qa_data:
    logger.error("No JSON files loaded. Please upload datasets to /data/")
    raise ValueError("No JSON files loaded. Please upload datasets to /data/")

# Extract documents and metadata
documents = [f"Question: {item['question']}\nAnswer: {item['answer']}" for item in qa_data]
metadata = [{"id": item["id"], "source": item["source"]} for item in qa_data]

# Set up embeddings and FAISS vector store
logger.info("Setting up embeddings and FAISS vector store...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = FAISS.from_texts(documents, embedding_model, metadatas=metadata)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Load fine-tuned model
model_name = "sravan837/ML_RAG_MODEl"  # Replace with your Hub model ID
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16
)
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, model_name)
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Set up text generation pipeline
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.6,
    top_p=0.85
)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Prompt template
prompt_template = """Based on the following Q&A pairs, provide a concise and accurate answer to the user's question. Use only the most relevant information from the context and avoid adding unnecessary details or multiple answers unless explicitly requested.

Retrieved Context:
{context}

User Question: {question}

Answer: """
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Set up RAG pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)
logger.info("RAG pipeline initialized")

# Query function
def query_rag(question, show_retrieval=False):
    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"].strip()
        sources = [doc.metadata["source"] for doc in result["source_documents"]]
        context = [doc.page_content for doc in result["source_documents"]]
        logger.info(f"Processed question: {question}, Answer: {answer}")
        
        output = f"**Answer**: {answer}"
        if show_retrieval:
            output += f"\n\n**Retrieval Details**:\n- **Retrieved Context**:\n"
            for i, ctx in enumerate(context, 1):
                output += f"  {i}. {ctx}\n"
            output += f"- **Sources**: {', '.join(sources)}"
        return output
    except Exception as e:
        logger.error(f"Error processing question '{question}': {str(e)}")
        return f"Error: Unable to process question. Please try again or rephrase."

# Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# Machine Learning Q&A RAG Model")
    gr.Markdown("Ask questions about machine learning based on 7,000 Q&A pairs. Supports short or full-form questions.")
    with gr.Row():
        question_input = gr.Textbox(label="Enter your question (e.g., 'What's gradient boosting?')", lines=2)
    with gr.Row():
        show_retrieval = gr.Checkbox(label="Show Retrieval Details (Context and Sources)", value=False)
    output = gr.Markdown(label="Answer")
    submit_button = gr.Button("Submit")
    submit_button.click(
        fn=query_rag,
        inputs=[question_input, show_retrieval],
        outputs=output
    )

# Launch
if __name__ == "__main__":
    iface.launch()