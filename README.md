# Knowledge Graph QA System

This repository contains the codebase developed for the master's thesis **"Knowledge Graphs for Retrieval-Augmented Generation"**, authored by *Hanna Jarlaczyńska* at *AGH University of Krakow*.


## Overview

This project implements a question-answering system using:
- Knowledge graphs built from document collections
- LLM-based entity extraction and graph transformation
- Hybrid retrieval combining vector search and graph traversal

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

See the documentation in each module for more details.