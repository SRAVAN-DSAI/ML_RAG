# ML Q&A RAG System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

This repository implements a **Retrieval-Augmented Generation (RAG)** pipeline for answering machine learning-related questions. The system uses a fine-tuned `facebook/opt-1.3b` model with LoRA adapters, a FAISS vector store for retrieval, and a Gradio interface for interactive Q&A. It leverages 7,000 Q&A pairs from 14 JSON datasets (`ml_qa_synthetic_set_*.json`) to provide accurate, context-aware responses. The project is deployed on **Hugging Face Spaces** (free CPU Basic tier) and includes metrics calculation (BLEU, ROUGE, semantic similarity, precision, recall, F1) in Google Colab.

### Key Features
- **Model**: Fine-tuned `facebook/opt-1.3b` with 8-bit quantization (LoRA adapters, trained for 1428/1689 steps, losses 0.8630/0.9584).
- **Retrieval**: FAISS vector store with `sentence-transformers/all-mpnet-base-v2` embeddings, retrieving top-3 relevant documents.
- **Interface**: Gradio dashboard displaying answers by default, with a toggle for retrieval details (context and sources).
- **Metrics**: Evaluates generation (BLEU, ROUGE, semantic similarity) and retrieval (precision, recall, F1) on a test set.
- **Deployment**: Hosted on Hugging Face Spaces for public access.
- **Environment**: Supports free tiers (Colab T4 GPU for metrics, Spaces CPU Basic for deployment).

### Use Case
Ideal for students, researchers, and ML enthusiasts seeking quick, accurate answers to machine learning questions (e.g., "What’s gradient boosting?"). Suitable for portfolio projects to showcase ML engineering skills.

## Repository Structure
- `app.py`: Gradio app for Hugging Face Spaces, combining dataset loading, FAISS setup, RAG pipeline, and interface.
- `requirements.txt`: Python dependencies for deployment.
- `data/`: Contains 14 JSON datasets (`ml_qa_synthetic_set_*.json`, 7000 Q&A pairs, 4500 training, 501 validation).
- `cell_7.py`: Metrics calculation script for Colab (BLEU, ROUGE, semantic similarity, precision, recall, F1).

## Installation and Setup

### Prerequisites
- **Hugging Face Account**: Sign up at [huggingface.co](https://huggingface.co) (free tier sufficient).[](https://github.com/SRAVAN-DSAI)
- **Colab**: Free T4 GPU (~16GB VRAM) for metrics calculation.
- **Git**: For cloning and pushing files (optional Git LFS for large model files).
- **Datasets**: 14 JSON files in `data/` (upload to Spaces or use local paths).
- **Model**: Fine-tuned model at `your-username/opt-1.3b-finetuned` on Hugging Face Hub or local checkpoint (`/content/finetune_opt_1.3b/final_checkpoint`).

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
faiss-gpu>=1.7.2
sentence-transformers>=2.2.2
evaluate>=0.4.0
accelerate>=0.20.0
bitsandbytes>=0.43.3
sentencepiece>=0.1.99
```

For Spaces, `faiss-gpu` is replaced by `faiss-cpu` (implicit in `langchain`).

## Deployment on Hugging Face Spaces
Deploy the Gradio app on Hugging Face Spaces (free CPU Basic, 2-core, 16GB RAM). Total time: ~11–21 minutes.

1. **Create a Space** (~1–2 minutes):
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click “Create new Space.”
   - Set:
     - **Name**: e.g., `ml-qa-rag-demo`.
     - **SDK**: Gradio.
     - **Hardware**: CPU Basic (free).
     - **Visibility**: Public (for sharing) or Private.
   - Create to get a URL: `https://huggingface.co/spaces/your-username/ml-qa-rag-demo`.

2. **Upload Files** (~3–5 minutes):
   - **Via UI**:
     - In the “Files” tab, create `requirements.txt` and `app.py` (copy from this repo).
     - Upload 14 JSON datasets to `data/` folder.
   - **Via Git**:
     ```bash
     git clone https://huggingface.co/spaces/your-username/ml-qa-rag-demo
     cd ml-qa-rag-demo
     cp path/to/ML_RAG/requirements.txt .
     cp path/to/ML_RAG/app.py .
     mkdir data && cp path/to/ML_RAG/data/*.json data/
     git add .
     git commit -m "Add app files and datasets"
     git push
     ```

3. **Model Setup** (~1 minute):
   - Use `your-username/opt-1.3b-finetuned` in `app.py` (line: `model_name = "your-username/opt-1.3b-finetuned"`).
   - If using a local checkpoint, upload to `/model/` in the Space and update `app.py` to load from `/model/`. Use Git LFS for large files:
     ```bash
     git lfs install
     git lfs track "*.bin"
     git add model/ .gitattributes
     git commit -m "Add model files"
     git push
     ```

4. **Build and Launch** (~5–10 minutes):
   - Spaces auto-installs dependencies and launches `app.py`.
   - Monitor the “App” tab for build logs. If errors occur, check `requirements.txt` or dataset paths (`/data/`).

5. **Test** (~1–2 minutes):
   - Open the Space URL.
   - Enter a question (e.g., “What’s gradient boosting?”).
   - Verify the answer and toggle retrieval details (context and sources).

### Troubleshooting
- **Build Fails**: Ensure `requirements.txt` matches dependencies. Check logs for missing packages or incorrect `app.py` paths.
- **Dataset Errors**: Verify all 14 JSON files are in `data/`. Update `app.py` if using a different path.
- **Memory Issues**: Use Hub model to reduce storage. Ensure 8-bit quantization (`load_in_8bit=True`) in `app.py`.
- **Slow Response**: Reduce `max_new_tokens=30` in `app.py` or upgrade to paid GPU tier (T4).

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
   - **Cell 3**: Load 14 JSON datasets (4500 training, 501 validation).
   - **Cell 4**: Set up FAISS (`k=3`).
   - **Cell 6**: Load model (`your-username/opt-1.3b-finetuned`, `max_new_tokens=50`, no `max_length`).
   - **Cell 7**: Run metrics (`cell_7.py`) with 3 test questions, no reranking, `batch_size=2`, `torch.cuda.empty_cache()`.
   - Skip **Cell 5** to avoid retraining.

3. **Imports for `cell_7.py`**:
   ```python
   from evaluate import load
   from sentence_transformers import SentenceTransformer, util
   from sklearn.metrics import precision_score, recall_score, f1_score
   from datasets import Dataset
   import torch
   ```

4. **Sample Metrics** (example, actual values depend on data):
   - Retrieval: `precision: 0.85`, `recall: 0.80`, `f1: 0.82`.
   - Generation: `bleu: 0.25`, `rouge1: 0.45`, `rougeL: 0.40`, `semantic_similarity: 0.90`.

### Issue Resolutions
- **Metrics Runtime**: Reduced from ~10 minutes to ~1–2 minutes by using 3 test questions, disabling reranking, batching, and caching embeddings.
- **Crashes**: Prevented by `torch.cuda.empty_cache()`, `max_new_tokens=50`, and smaller test set.
- **Warnings**:
  - Fixed `max_new_tokens=256`/`max_length=21` conflict by removing `max_length`.
  - `bitsandbytes` casting warning (torch.bfloat16 to float16) is harmless.

## Usage
- **Gradio App**:
  - Access at `https://huggingface.co/spaces/your-username/ml-qa-rag-demo`.
  - Enter questions (e.g., “Explain gradient descent”).
  - Toggle “Show Retrieval Details” for context and sources.
- **Colab Metrics**:
  - Run `cell_7.py` to evaluate performance.
  - Adjust test size or `max_new_tokens` for trade-offs between accuracy and speed.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a branch (`git checkout -b feature/your-feature`).
3. Add improvements (e.g., new datasets, metrics, or UI features).
4. Submit a pull request with a clear description.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **Hugging Face**: For hosting the model and Spaces platform.[](https://github.com/SRAVAN-DSAI)
- **LangChain**: For RAG pipeline and FAISS integration.
- **Sentence Transformers**: For embeddings (`all-mpnet-base-v2`).
- **SRAVAN-DSAI**: Built by Sravan Kodari, a Data Science enthusiast passionate about AI and ML.[](https://github.com/SRAVAN-DSAI)

## Contact
For issues or suggestions, open a GitHub issue or contact [Sravan Kodari](https://www.linkedin.com/in/sravan-kodari-943654210/).[](https://github.com/SRAVAN-DSAI)

---