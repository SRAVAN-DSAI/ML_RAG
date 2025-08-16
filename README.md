# ML Q&A Chatbot

## Overview

This Hugging Face Space hosts an interactive **Machine Learning Q&A Chatbot** powered by a Retrieval-Augmented Generation (RAG) pipeline. The system uses a fine-tuned `facebook/opt-1.3b` model with LoRA adapters, a FAISS vector store for retrieval, and a Gradio interface for user interaction. It leverages 7,000 Q&A pairs from 14 JSON datasets (`ml_qa_synthetic_set_*.json`) to provide accurate, context-aware answers to machine learning questions. The project is deployed on **Hugging Face Spaces** (free CPU Basic tier) and supports metrics calculation (BLEU, ROUGE, semantic similarity, precision, recall, F1) in Google Colab.

### Key Features
- **Model**: Fine-tuned `facebook/opt-1.3b` with 8-bit quantization (LoRA adapters, trained for 1428/1689 steps, losses 0.803300/0.936291).
- **Retrieval**: FAISS vector store with `sentence-transformers/all-mpnet-base-v2` embeddings, retrieving top-3 relevant documents.
- **Interface**: Gradio dashboard displaying answers by default, with a toggle for retrieval details (context and sources).
- **Deployment**: Hosted at `https://huggingface.co/spaces/sravan837/ML_CHATBOT` for public access.
- **Environment**: Supports free tiers (Colab T4 GPU for metrics, Spaces CPU Basic for deployment).

### Use Case
Ideal for students, researchers, and ML enthusiasts seeking quick answers to machine learning questions (e.g., "What’s gradient boosting?"). Perfect for showcasing ML engineering skills in a portfolio.

## Repository Structure
- `app.py`: Gradio app for Hugging Face Spaces, combining dataset loading, FAISS setup, RAG pipeline, and interface.
- `requirements.txt`: Python dependencies for deployment.
- `data/`: Contains 14 JSON datasets (`ml_qa_synthetic_set_*.json`, 7,000 Q&A pairs, 4,500 training, 501 validation).

## Installation and Setup

### Prerequisites
- **Hugging Face Account**: Sign up at [huggingface.co](https://huggingface.co) (free tier sufficient).
- **Colab**: Free T4 GPU (~16GB VRAM) for metrics calculation.
- **Git**: For cloning and pushing files (optional Git LFS for large model files).
- **Datasets**: 14 JSON files in `data/` (upload to Spaces or use local paths).
- **Model**: Fine-tuned model at `sravan837/opt-1.3b-finetuned` on Hugging Face Hub or local checkpoint (`/content/finetune_opt_1.3b/final_checkpoint`).

### Dependencies
Install required packages listed in `requirements.txt`:
```text
torch>=2.0.0
langchain>=0.1.46
langchain-community>=0.0.38
langchain-huggingface>=0.0.3
transformers>=4.30.0
datasets>=2.14.0
peft>=0.10.0
trl>=0.7.0
gradio>=4.0.0
faiss-cpu>=1.7.2
sentence-transformers>=2.2.2
evaluate>=0.4.0
accelerate>=0.20.0
bitsandbytes>=0.43.3
sentencepiece>=0.1.99
```

**Note**: For Spaces, `faiss-gpu` is replaced by `faiss-cpu` (implicit in `langchain`).

## Deployment on Hugging Face Spaces
Deploy the Gradio app at `https://huggingface.co/spaces/sravan837/ML_CHATBOT` using the free CPU Basic tier (2-core, 16GB RAM). Total time: ~11–21 minutes.

1. **Verify Space** (~1 minute):
   - Ensure the Space exists at [huggingface.co/spaces/sravan837/ML_CHATBOT](https://huggingface.co/spaces/sravan837/ML_CHATBOT). If not, create it:
     - Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click “Create new Space.”
     - Set:
       - **Name**: `ML_CHATBOT`.
       - **SDK**: Gradio.
       - **Hardware**: CPU Basic (free).
       - **Visibility**: Public (for sharing) or Private.
     - Create to get the URL: `https://huggingface.co/spaces/sravan837/ML_CHATBOT`.

2. **Upload Files** (~3–5 minutes):
   - **Via UI**:
     - In the Space’s “Files” tab, create `requirements.txt` and `app.py` (copy from [GitHub](https://github.com/SRAVAN-DSAI/ML_RAG)).
     - Upload 14 JSON datasets to `data/` folder.
   - **Via Git**:
     ```bash
     git clone https://huggingface.co/spaces/sravan837/ML_CHATBOT
     cd ML_CHATBOT
     cp path/to/ML_RAG/requirements.txt .
     cp path/to/ML_RAG/app.py .
     mkdir data && cp path/to/ML_RAG/data/*.json data/
     git add .
     git commit -m "Add app files and datasets"
     git push
     ```

3. **Model Setup** (~1 minute):
   - Update `app.py` to use `sravan837/opt-1.3b-finetuned` (line: `model_name = "sravan837/opt-1.3b-finetuned"`).
   - If using a local checkpoint, upload to `/model/` in the Space and modify `app.py` to load from `/model/`. Use Git LFS for large files:
     ```bash
     git lfs install
     git lfs track "*.bin"
     git add model/ .gitattributes
     git commit -m "Add model files"
     git push
     ```

4. **Build and Launch** (~5–10 minutes):
   - Spaces auto-installs dependencies from `requirements.txt` and launches `app.py`.
   - Monitor the “App” tab for build logs. If errors occur, check `requirements.txt` or dataset paths (`/data/`).

5. **Test** (~1–2 minutes):
   - Open [huggingface.co/spaces/sravan837/ML_CHATBOT](https://huggingface.co/spaces/sravan837/ML_CHATBOT).
   - Enter a question (e.g., “Explain gradient descent”).
   - Verify the answer and toggle “Show Retrieval Details” for context and sources.

### Troubleshooting
- **Build Fails**: Verify `requirements.txt` dependencies. Check logs for missing packages or incorrect `app.py` paths (e.g., `/data/`).
- **Dataset Errors**: Ensure all 14 JSON files are in `data/`. Update `app.py` if using a different path.
- **Memory Issues**: Use Hub model to reduce storage. Confirm 8-bit quantization (`load_in_8bit=True`) in `app.py`.
- **Slow Response**: Reduce `max_new_tokens=30` in `app.py` or upgrade to paid GPU tier (T4).[](https://huggingface.co/docs/hub/main/en/spaces-overview)

## Metrics Calculation in Colab
Evaluate the RAG system (BLEU, ROUGE, semantic similarity, precision, recall, F1) using `cell_7.py` in Colab T4 GPU. Optimized to prevent crashes and reduce runtime (~1–2 minutes).

1. **Setup Colab** (~2–3 minutes):
   - Open a Colab notebook with T4 GPU.
   - Install dependencies:
     ```bash
     !pip install torch langchain langchain-community langchain-huggingface transformers datasets peft trl gradio faiss-gpu sentence-transformers evaluate accelerate bitsandbytes sentencepiece
     ```

2. **Run Cells** (~1–2 minutes):
   - **Cell 1**: Install dependencies (above).
   - **Cell 2**: Load datasets and logging setup.
   - **Cell 3**: Load 14 JSON datasets (4,500 training, 501 validation).
   - **Cell 4**: Set up FAISS (`k=3`).
   - **Cell 6**: Load model (`sravan837/opt-1.3b-finetuned`, `max_new_tokens=50`, no `max_length`).
   - Skip **Cell 5** to avoid retraining.


## Usage
- **Gradio App**:
  - Access at [huggingface.co/spaces/sravan837/ML_CHATBOT](https://huggingface.co/spaces/sravan837/ML_CHATBOT).
  - Enter questions (e.g., “What’s gradient boosting?”).
  - Toggle “Show Retrieval Details” for context and sources.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository: [github.com/SRAVAN-DSAI/ML_RAG](https://github.com/SRAVAN-DSAI/ML_RAG).
2. Create a branch (`git checkout -b feature/your-feature`).
3. Add improvements (e.g., new datasets, metrics, or UI features).
4. Submit a pull request with a clear description.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **Hugging Face**: For hosting the model and Spaces platform.[](https://huggingface.co/)[](https://huggingface.co/docs/hub/main/en/spaces-overview)
- **LangChain**: For RAG pipeline and FAISS integration.
- **Sentence Transformers**: For embeddings (`all-mpnet-base-v2`).
- **SRAVAN-DSAI**: Built by Sravan Kodari, a Data Science enthusiast passionate about AI and ML.

## Contact
For issues or suggestions, open a GitHub issue at [github.com/SRAVAN-DSAI/ML_RAG](https://github.com/SRAVAN-DSAI/ML_RAG) or contact [Sravan Kodari](https://www.linkedin.com/in/sravan-kodari/).

---