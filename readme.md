# Vector Space Model for Information Retrieval

This repository contains the implementation of a Vector Space Model (VSM) for information retrieval. The VSM allows users to search a collection of research papers using specified queries and retrieve relevant documents based on cosine similarity.

## Features

- Preprocessing Pipeline:
  - Tokenization
  - Case Folding
  - Stop-words Removal
  - Stemming

- Query Processing:
  - Query Parsing
  - Evaluation of Query Cost
  - Execution for Document Retrieval

- Vector Space Model:
  - Construction of Feature Space using TF-IDF
  - Cosine Similarity Calculation

- Graphical User Interface (GUI) using Tkinter:
  - Query Entry Field
  - Search Button
  - Display of Search Results
  - Clear Output Button

## How to Use

1. Clone the repository to your local machine.
2. Install the required dependencies: Python 3.x, NLTK (Natural Language Toolkit), Tkinter.
3. Run the Python script `main.py`.
4. Enter your query in the provided field and click 'Search' to retrieve relevant documents. Use the 'Clear Output' button to clear the search results.

## File Descriptions

- `main.py`: Main Python script containing the implementation.
- `Stopword-List.txt`: Text file containing a list of stopwords used for preprocessing.
- `ResearchPapers/`: Directory containing sample research papers (text files) for indexing and querying.

## Author

This GUI application was developed by Abdul Rehman.

## Repository

[View on GitHub](https://github.com/bawanyabdulrehman/Vector-Space-Model-for-IR.git)