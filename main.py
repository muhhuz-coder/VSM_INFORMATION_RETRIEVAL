import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
import os
import re
import math
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK data
nltk.download('punkt')

def preprocess_text(text):
    """
    Preprocesses text by tokenizing, stemming, removing stopwords,
    casefolding, filtering tokens, preserving special cases, and
    handling punctuation and digits.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        list: A list of preprocessed tokens.
    """
    # Load stopwords
    stop_words = load_stopwords('Stopword-List.txt')

    # Create stemmer for reducing words to their root forms
    stemmer = PorterStemmer()

    # Define regular expressions for email and URL patterns
    email_regex = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    url_regex = r'https?://[^\s]+'

    # Define a function to handle punctuation and digits
    def clean_token(token):
        if re.match(email_regex, token) or re.match(url_regex, token):
            return token  # Preserve emails and URLs
        else:
            return re.sub(r'[^a-zA-Z0-9]', '', token)  # Remove all non-alphanumeric characters

    # Split text into words (considering special cases)
    tokens = []
    for word in re.findall(r'(?:' + email_regex + r'|' + url_regex + r'|\S+)', text):
        tokens.append(clean_token(word))

    # Preprocess tokens:
    processed_tokens = []
    for token in tokens:
        # Convert to lowercase (excluding emails and URLs)
        if not re.match(email_regex, token) and not re.match(url_regex, token):
            token = token.lower()
            token = re.sub(r'\d', '', token) # Drop digits from the token

        # Retain tokens containing at least one alphabetic character
        if any(c.isalpha() for c in token):
            if len(token) > 2:  # Exclude short tokens
                stemmed_token = stemmer.stem(token)  # Reduce word to root form
                if stemmed_token not in stop_words:  # Exclude stopwords
                    processed_tokens.append(stemmed_token)

    return processed_tokens

def load_stopwords(filename):
    """
    Loads stopwords from a file.

    Args:
    - filename (str): The path to the file containing stopwords.

    Returns:
    - set: A set of stopwords.
    """
    stopwords = set()
    with open(filename, 'r') as file:
        for line in file:
            stopwords.add(line.strip())
    return stopwords

def load_docs(directory):
    """
    Loads documents from a directory.

    Args:
    - directory (str): The path to the directory containing documents.

    Returns:
    - dict: A dictionary where keys are document IDs and values are documents.
    """
    docs = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            doc_id = int(os.path.splitext(filename)[0])  # Extract document ID from filename
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                docs[doc_id] = file.read()
    return docs

def build_index(docs, index_file="index.txt"):
    """
    Builds an inverted index from a collection of documents.

    Args:
    - docs (dict): A dictionary where keys are document IDs and values are documents.
    - index_file (str): The file to store the index.

    Returns:
    - dict: The inverted index.
    """
    if os.path.exists(index_file):
        index = load_index(index_file)
        return index

    index = {}

    for doc_id, document in docs.items():
        # Tokenize and preprocess document
        tokens = preprocess_text(document)

        # Compute term frequencies for the document
        term_freq = {}
        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1

        # Update index with document's terms and term frequencies
        for token, freq in term_freq.items():
            if token not in index:
                index[token] = {}  # Initialize a dictionary for the term
            index[token][doc_id] = freq  # Store term frequency for the document

    # Sort the index by keys (terms) alphabetically
    index = {k: index[k] for k in sorted(index)}

    # Save index to file
    save_index(index, index_file)

    return index

def save_index(index, index_file):
    """
    Saves the index to a text file.

    Args:
    - index (dict): The index to be saved.
    - index_file (str): The file to save the index to.
    """
    with open(index_file, 'w') as f:
        for term, postings in index.items():
            f.write(term + ':')
            for doc_id, freq in postings.items():
                f.write(f'({doc_id},{freq})')
            f.write('\n')

def load_index(index_file):
    """
    Loads the index from a text file.

    Args:
    - index_file (str): The file to load the index from.

    Returns:
    - dict: The loaded index.
    """
    index = {}
    with open(index_file, 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) == 2:  # Ensure the line has two parts
                term = parts[0]
                postings_str = parts[1]
                postings = {}
                if postings_str:  # Check if there are postings
                    pairs = postings_str.split(')(')
                    for pair in pairs:
                        doc_id, freq = pair.strip('()').split(',')
                        postings[int(doc_id)] = int(freq)  # Convert doc_id and freq to integers
                index[term] = postings
    return index

def calculate_tf_idf(term, doc_id, index, docs, query_term_freq=None):
    """
    Calculate TF-IDF score for a term in a document or query.

    Args:
    - term (str): The term for which TF-IDF score is calculated.
    - doc_id (int or None): The ID of the document. If None, indicates the term is from the query.
    - index (dict): The inverted index.
    - docs (dict): A dictionary of documents.
    - query_term_freq (int or None): The frequency of the term in the query. None if calculating for a document.

    Returns:
    - float: The TF-IDF score.
    """
    if doc_id is None:
        # Calculate term frequency (TF) based on the number of occurrences of the term in the query
        doc_freq = query_term_freq
    else:
        doc_freq = index[term].get(doc_id, 0)

    tf = doc_freq
    idf = math.log(len(docs) / len(index[term]), 10) if len(index[term]) != 0 else 0  # Handle division by zero
    return tf * idf

def calculate_query_vector(query, index, docs):
    """
    Calculate TF-IDF vector for the query.

    Args:
    - query (str): The query string.
    - index (dict): The inverted index.
    - docs (dict): A dictionary of documents.

    Returns:
    - np.array: The TF-IDF vector for the query.
    """
    vector = np.zeros(len(index))  # Initialize a vector with zeros
    tokens = preprocess_text(query)
    query_term_freq = {term: tokens.count(term) for term in set(tokens)}  # Calculate term frequency in the query
    for i, term in enumerate(index.keys()):
        if term in tokens:
            vector[i] = calculate_tf_idf(term, None, index, docs, query_term_freq[term])
    return vector

def calculate_document_vectors(index, docs):
    """
    Calculate TF-IDF vectors for all documents.

    Args:
    - index (dict): The inverted index.
    - docs (dict): A dictionary of documents.

    Returns:
    - dict: A dictionary where keys are document IDs and values are TF-IDF vectors.
    """
    doc_vectors = {}
    for doc_id, _ in docs.items():
        vector = np.zeros(len(index))  # Initialize a vector with zeros
        for i, term in enumerate(index.keys()):
            vector[i] = calculate_tf_idf(term, doc_id, index, docs)
        doc_vectors[doc_id] = vector
    return doc_vectors

def cosine_similarity(vector1, vector2):
    """
    Calculate the cosine similarity between two vectors.

    Args:
    - vector1 (np.array): The first vector.
    - vector2 (np.array): The second vector.

    Returns:
    - float: The cosine similarity between the two vectors.
    """
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    if norm1 != 0 and norm2 != 0:
        similarity = dot_product / (norm1 * norm2)
    else:
        similarity = 0  # Handle division by zero
    return similarity

def perform_search(query, index, docs, alpha=0.05):
    """
    Perform a search using the Vector Space Model.

    Args:
    - query (str): The query string.
    - index (dict): The inverted index.
    - docs (dict): A dictionary of documents.
    - alpha (float): The threshold to filter results.

    Returns:
    - list: A ranked list of document IDs based on relevance to the query.
    """
    query_vector = calculate_query_vector(query, index, docs)
    doc_vectors = calculate_document_vectors(index, docs)

    # Calculate cosine similarity between query vector and document vectors
    similarities = {doc_id: cosine_similarity(query_vector, doc_vector) for doc_id, doc_vector in doc_vectors.items()}

    # Filter documents based on threshold alpha and extract relevant document IDs
    relevant_docs = [doc_id for doc_id, similarity in similarities.items() if similarity >= alpha]

    # Sort relevant documents by descending order of similarity
    ranked_docs = sorted(relevant_docs, key=lambda doc_id: similarities.get(doc_id, 0), reverse=True)

    return ranked_docs

# Define the main GUI function
def run_gui():
    def search(event=None):
        query = query_entry.get()
        if query:
            # Perform search
            ranked_docs = perform_search(query, index, docs)
            # Display query
            results_text.insert(tk.END, "Query: " + query + "\n")
            if ranked_docs:
                # Display results
                results_text.insert(tk.END, "Relevant Document IDs:\n")
                for doc_id in ranked_docs:
                    results_text.insert(tk.END, f"{doc_id}\n")
            else:
                results_text.insert(tk.END, "No results found for the query.\n")
            # Add separation line
            results_text.insert(tk.END, "-" * 50 + "\n")
        else:
            messagebox.showwarning("Empty Query", "Please enter a query.")
        query_entry.delete(0, tk.END)
        query_entry.insert(0, "Enter your query here")
        query_entry.icursor(0)  # Move cursor to start
        query_entry.focus()  # Move focus to the entry widget

    def clear_output():
        results_text.delete('1.0', tk.END)

    def on_entry_click(event):
        if query_entry.get() == "Enter your query here":
            query_entry.delete(0, tk.END)

    def on_focus_out(event):
        if query_entry.get() == "":
            query_entry.insert(0, "Enter your query here")

    # Load documents
    docs = load_docs('ResearchPapers')

    # Build index
    index = build_index(docs)

    # Create GUI
    root = tk.Tk()
    root.title("Vector Space Model")
    root.geometry("600x400")

    input_frame = tk.Frame(root)
    input_frame.pack(pady=10)

    query_entry = tk.Entry(input_frame, width=50)
    query_entry.pack(side=tk.LEFT)
    query_entry.insert(0, "Enter your query here")
    query_entry.bind("<FocusIn>", on_entry_click)
    query_entry.bind("<FocusOut>", on_focus_out)
    query_entry.focus_set()  # Set focus to the query entry

    search_button = tk.Button(input_frame, text="Search", command=search)
    search_button.pack(side=tk.LEFT)
    root.bind('<Return>', search)

    clear_button = tk.Button(root, text="Clear Output", command=clear_output)
    clear_button.pack(side=tk.BOTTOM, pady=10)

    results_label = tk.Label(root, text="Search Results:")
    results_label.pack()

    results_text = scrolledtext.ScrolledText(root, width=50, height=10, wrap=tk.WORD)
    results_text.pack(expand=True, fill='both')

    root.mainloop()

if __name__ == "__main__":
    run_gui()
