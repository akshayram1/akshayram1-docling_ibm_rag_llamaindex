# RAG over Excel with Dockling & Llama-3.2

This project is a Streamlit-based application that enables **Retrieval-Augmented Generation (RAG)** over Excel files. The app leverages **Dockling**, **Llama-3.2**, and Hugging Face embeddings to index and query document data efficiently.

## Features
- **Upload Excel Files**: Users can upload `.xlsx` or `.xls` files.
- **Excel Data Preview**: Displays a preview of the uploaded Excel file.
- **Intelligent Querying**: Uses a custom prompt template for precise answers.
- **Interactive Chat**: Chat-based interface for querying document content.
- **Clear Chat History**: Reset the chat history and context with a single click.

## Tech Stack
- **Backend**: Python, Streamlit
- **Machine Learning Models**:
  - **LLM**: Llama-3.2 (via Ollama)
  - **Embeddings**: Hugging Face `BAAI/bge-large-en-v1.5`
- **Libraries**:
  - `llama_index`
  - `pandas`
  - `streamlit`

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your_username/rag-excel-dockling.git
   cd rag-excel-dockling
   ```

2. **Install Dependencies**
   Make sure you have Python 3.8 or later installed. Then, run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Upload a File**: Navigate to the sidebar and upload an Excel file.
2. **Indexing**: The application will process and index the file.
3. **Query**: Ask questions in the chat interface, and the system will provide answers based on the file's content.
4. **Clear Chat**: Use the `Clear ↺` button to reset the chat.

## Screenshots

### Home Screen
![Home Screen](home_screen.png)

### Excel Preview
![Excel Preview](excelsheet.png)

### Chat Interface
![Chat Interface](chat.png)

## Customization

- **Prompt Template**: The prompt used for querying can be updated in the code:
  ```python
  qa_prompt_tmpl_str = (
      "Context information is below.\n"
      "---------------------\n"
      "{context_str}\n"
      "---------------------\n"
      "Given the context information above I want you to think step by step to answer the query in a highly precise and crisp manner focused on the final answer, incase case you don't know the answer say 'I don't know!'.\n"
      "Query: {query_str}\n"
      "Answer: "
  )
  ```

- **Embedding Model**: Change the embedding model by modifying this line:
  ```python
  embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
  ```

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments
- [Llama Index](https://github.com/jerryjliu/llama_index)
- [Streamlit](https://streamlit.io)
- [Hugging Face](https://huggingface.co)

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

---

Feel free to modify this README.md file to better suit your project structure and needs.
# docling_ibm_rag_llamaindex
# docling_ibm_rag_llamaindex
