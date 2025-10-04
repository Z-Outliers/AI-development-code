# AI Development for NASA Space Apps Challenge

A comprehensive AI-powered system for processing and querying scientific research documents using two complementary approaches: **Multimodal RAG** and **Knowledge Graph RAG**.

---

## Project Overview

This project implements two advanced Retrieval-Augmented Generation (RAG) systems designed to extract, store, and query information from scientific research papers:

1. **Multimodal RAG Pipeline** - Processes PDFs with text, tables, and images
2. **Knowledge Graph RAG** - Builds structured knowledge graphs from medical research

Both systems run **100% locally** using open-source tools with **no API costs** (except optional Google Gemini for knowledge graph extraction).

---

## Project Structure

```bash
AI-Code/
├── clean_final_rag_v3.ipynb          # Multimodal RAG implementation
├── knowledge-graph-construction.ipynb # Knowledge Graph RAG implementation
└── README.md                          # This file
```

---

## System 1: Multimodal RAG Pipeline

### Overview

A production-ready RAG system that processes multiple PDFs simultaneously, extracting and indexing text, tables, and images for semantic search and question answering.

### Key Features

- **Multi-PDF Processing** - Batch process multiple research papers with source tracking  
- **Multimodal Extraction** - Extracts text, tables, and embedded images from PDFs  
- **Intelligent Summarization** - Generates summaries for better retrieval accuracy  
- **Dual Storage Architecture** - Summaries in vector DB, raw content in document store  
- **Source Attribution** - Track which document answers come from  
- **Incremental Updates** - Add new PDFs without reprocessing everything  
- **Export to CSV** - Export embeddings and content for backend integration  

### Tech Stack

- **LangChain** - Orchestration framework
- **Ollama** - Local LLM inference (Gemma, Llama models)
- **Unstructured** - Advanced PDF parsing
- **ChromaDB** - Vector database for embeddings
- **MultiVectorRetriever** - Efficient summary-based retrieval

### Workflow

```bash
1. Load PDFs → 2. Parse (text/tables/images) → 3. Summarize each element
    ↓
4. Store summaries in vector DB → 5. Store raw content separately
    ↓
6. Query → 7. Retrieve relevant summaries → 8. Return full content → 9. Generate answer
```

### Usage Example

```python
# Process multiple PDFs
pdf_paths = ["paper1.pdf", "paper2.pdf", "paper3.pdf"]
all_elements_by_source = process_multiple_pdfs(pdf_paths)

# Query across all documents
query("What are the main findings in the research?")

# Export for backend integration
export_complete_dataset("nasa_rag_export")
```

### Output Files

- `nasa_rag_export_vectors.csv` - Vector embeddings and summaries
- `nasa_rag_export_content.csv` - Raw document content

---

## System 2: Knowledge Graph RAG

### Overview

Builds structured knowledge graphs from medical research papers, extracting entities (diseases, treatments, concepts) and relationships to enable graph-based querying and visualization.

### Key Features

- **Automated Entity Extraction** - LLM identifies medical concepts, diseases, treatments  
- **Relationship Mapping** - Discovers connections between entities  
- **Graph Visualization** - Interactive exploration using yFiles widgets  
- **Query-Driven Subgraphs** - Extract relevant portions of the graph based on queries  
- **Source Traceability** - Link entities back to source documents  
- **Structured Section Parsing** - Preserves paper structure (Abstract, Methods, Results)  

### Tech Stack

- **Neo4j** - Graph database for knowledge storage
- **Google Gemini 2.5 Flash** - LLM for entity/relationship extraction
- **LangChain** - Document processing and graph transformers
- **yFiles** - Interactive graph visualization
- **LangSmith** - LLM observability (optional)

### Workflow

```bash
1. Load JSON papers → 2. Split into chunks → 3. Extract entities & relationships
    ↓
4. Build graph documents → 5. Store in Neo4j → 6. Query & visualize
```

### Usage Example

```python
# Load research papers
raw_documents = load_publication_documents(json_directory="papers", max_docs=5)

# Build knowledge graph
graph_documents = llm_transformer.convert_to_graph_documents(documents)
graph.add_graph_documents(graph_documents)

# Query-based visualization
visualize_query_subgraph("What research exists about liver dysfunction?", top_k=10)
```

### Graph Visualization

The system provides interactive graph visualizations showing:

- **Nodes** - Entities (diseases, treatments, concepts)
- **Edges** - Relationships (causes, treats, associates with)
- **Context** - Clickable nodes with source document links

---

## Installation & Setup

### Prerequisites

1. **Python 3.10+**
2. **Ollama** (for Multimodal RAG)

   ```bash
   # Download from https://ollama.ai/download
   ollama serve
   ollama pull gemma3:4b
   ollama pull embeddinggemma:300m
   ollama pull llava:7b  # For image descriptions
   ```

3. **Neo4j Database** (for Knowledge Graph RAG)

   ```bash
   # Download from https://neo4j.com/download/
   # Or use Neo4j AuraDB (cloud)
   ```

4. **System Dependencies**

   ```bash
   # For PDF processing (Ubuntu/Debian)
   sudo apt-get install poppler-utils tesseract-ocr
   
   # For macOS
   brew install poppler tesseract
   ```

### Python Dependencies

```bash
# Multimodal RAG dependencies
pip install langchain langchain-community langchain-ollama
pip install chromadb
pip install "unstructured[all-docs]" python-magic
pip install pillow pdf2image pdfminer-six
pip install pandas openpyxl tabulate pytesseract

# Knowledge Graph RAG dependencies
pip install neo4j
pip install langchain-google-genai
pip install langchain-experimental
pip install yfiles-jupyter-graphs
pip install python-dotenv
```

### Environment Configuration

Create a `.env` file for Knowledge Graph RAG:

```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# API Keys
GOOGLE_API_KEY=your_google_api_key
LANGSMITH_API_KEY=your_langsmith_key  # Optional
LANGSMITH_PROJECT=your_project_name   # Optional
```

---

## Use Cases

### Multimodal RAG

- Research paper analysis and comparison
- Extracting data from tables and figures
- Finding specific information across multiple documents
- Generating summaries and insights
- Document categorization and filtering

### Knowledge Graph RAG

- Discovering relationships between medical concepts
- Clinical research exploration
- Drug-disease-symptom network analysis
- Visualizing research landscapes
- Identifying research gaps and connections

---

## Workflow Comparison

| Feature | Multimodal RAG | Knowledge Graph RAG |
|---------|---------------|---------------------|
| **Input Format** | PDF files | JSON papers |
| **Processing** | Text, tables, images | Text chunks |
| **Storage** | Vector embeddings | Graph database |
| **Retrieval** | Semantic similarity | Graph traversal |
| **Visualization** | Document excerpts | Interactive graphs |
| **Best For** | Direct Q&A, multimodal content | Relationship discovery, exploration |

---

## Getting Started

### Quick Start: Multimodal RAG

1. Place PDF files in `pdfs/` folder
2. Open `clean_final_rag_v3.ipynb`
3. Run all cells sequentially
4. Query your documents using `query("your question")`
5. Export results using `export_complete_dataset()`

### Quick Start: Knowledge Graph RAG

1. Place JSON papers in `papers/` folder
2. Configure `.env` with Neo4j credentials
3. Open `knowledge-graph-construction.ipynb`
4. Run all cells to build the graph
5. Visualize with `visualize_query_subgraph("your query")`

---

## Performance & Limitations

### Multimodal RAG

- **Processing Speed**: ~2-5 minutes per PDF (depends on size and complexity)
- **Memory Usage**: ~2-4GB RAM for typical workloads
- **Accuracy**: High for text, good for tables, variable for images

### Knowledge Graph RAG

- **Graph Building**: ~30-60 seconds per document chunk
- **Query Speed**: Sub-second for most graph queries
- **Scalability**: Handles 100s of documents efficiently

---

## Contributing

This project was developed for the NASA Space Apps Challenge. Contributions, improvements, and extensions are welcome!

---

## License

Open source - please check individual dependencies for their licenses.

---

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Models](https://ollama.ai/library)
- [Neo4j Graph Database](https://neo4j.com/)
- [Unstructured.io](https://unstructured.io/)
- [ChromaDB](https://www.trychroma.com/)

---

## Support

For questions or issues:

1. Check notebook cell documentation
2. Review error messages and troubleshooting guides
3. Ensure all dependencies are correctly installed
4. Verify Ollama/Neo4j services are running

---

**Built for NASA Space Apps Challenge**
