## **RerankRetriever**

### **Step 1: Input**

* Receive a **user question**.

---

### **Step 2: Structured Retrieval (SmartKGRetriever)**

#### 2.1 Entity Extraction

* Extract named entities from the question using `EntityExtractor`.

#### 2.2 Graph Querying for Candidates

For each extracted entity:

* Perform a **full-text search** on Neo4j entity index.
* Match documents (`Document`) that `MENTION` the entity.
* Optionally expand 1–2 hops via `RELATED_TO` to related entities and their mentioned documents.
* Collect and deduplicate up to 20 documents.

#### 2.3 Semantic Ranking

* Embed the question and candidate documents using `OpenAIEmbeddings`.
* Compute **cosine similarity** between question and each document.
* Sort candidates by similarity score.

#### 2.4 Smart Selection

* Select top-ranked document.
* Iteratively add others if:

  * Score is **above threshold** (e.g., > 0.3).
  * Score doesn’t **drop too much** (e.g., delta > 0.4 → stop).
* Ensure at least 2 docs if possible.

Output: list of selected document texts

---

### **Step 3: Standard Retrieval**


#### 3.1 Vector Search

* Use `Neo4jVector.similarity_search()` to retrieve top-k similar documents using hybrid similarity search.


➡️ Output: `standard_docs` (list of retrieved document texts)

---

### **Step 4: Merge Results**


---

### **Step 5: LLM-based Reranking**

#### 5.1 Prompt Construction

* Format prompt with:

  * The **question**.
  * Numbered list of `combined_docs`.

#### 5.2 LLM Selection

* Ask the LLM: “Which documents are most helpful for answering the question?”
* Parse returned list of selected document indices.

#### 5.3 Final Selection

* Return the selected documents based on LLM's response.

---

## **Final Output**

* List of the most relevant documents for the input question.
