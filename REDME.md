# Knowledge Graph QA System

This project implements a question-answering system using:
- Knowledge graphs built from document collections
- LLM-based entity extraction and graph transformation
- Hybrid retrieval combining vector search and graph traversal
- Chat history management for follow-up questions

## Setup

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - MacOS/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.sample` to `.env` and fill in your API keys and database credentials

## Usage

### Building the Knowledge Graph
```bash
python -m scripts.build_knowledge_graph
```

### Interactive QA
```bash
python -m scripts.interactive_qa
```

See the documentation in each module for more details.