# This little program shows how you can answer a question by looking
# up information in a PDF.  It does a “multi‑hop” search: you grab some
# text, ask a model to refine your query, grab more text, and finally
# ask a language model to put everything together.
#
# **Steps in the workflow**:
# 1. Read a PDF and pull out all of its text.
# 2. Chop the text into overlapping pieces so we can search inside it.
# 3. Turn those pieces into number vectors and store them in a FAISS index.
# 4. For a question, look up the most similar pieces and rank them.
# 5. (If doing more than one hop) re‑phrase the question based on what
#    you’ve already found and repeat the search.
# 6. Build a prompt with the gathered passages and have a causal
#    language model generate the final answer.
#
# The comments on each line below explain the code in simple terms.

from pypdf import PdfReader          # PDF reading helper class

def load_pdf(path):
    # make a reader object for the file at `path`
    reader = PdfReader(path)
    text = ""                       # start with empty text
    for page in reader.pages:       # go over every page in the PDF
        # get the text from that page and add a newline after it
        text += page.extract_text() + "\n"
    return text                     # give back all the text we collected

# actually load a specific PDF file into a variable
document_text = load_pdf("constitution.pdf")

def chunk_text(text, chunk_size=400, overlap=50):
    # split the big string `text` into smaller strings of roughly
    # `chunk_size` characters, allowing each piece to share
    # `overlap` characters with the previous piece
    chunks = []                     # list to hold the pieces
    start = 0                       # where to start the next slice
    while start < len(text):        # until we’ve covered everything
        end = start + chunk_size    # compute the end index for this slice
        chunks.append(text[start:end])  # take that slice and keep it
        # move start forward but keep the overlap
        start += chunk_size - overlap
    return chunks                   # return the list of chunks

# cut the document text into bite-sized chunks
chunks = chunk_text(document_text)

# bring in the libraries we need for encoding and indexing
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# load two small transformer models:
#   one for encoding queries,
#   one for encoding text passages
query_encoder = SentenceTransformer("facebook-dpr-question_encoder-single-nq-base")
passage_encoder = SentenceTransformer("facebook-dpr-ctx_encoder-single-nq-base")

# turn every text chunk into a numeric vector
passage_embeddings = passage_encoder.encode(chunks, convert_to_numpy=True)

# figure out how many dimensions each vector has
dimension = passage_embeddings.shape[1]
# make a FAISS index that will let us do inner‑product search
index = faiss.IndexFlatIP(dimension)
# put all of our passage vectors into the index
index.add(passage_embeddings)

def retrieve(query, top_k=15):
    # encode the query string into a vector
    query_vec = query_encoder.encode([query])
    # ask the index for the `top_k` chunks most like the query
    scores, indices = index.search(query_vec, top_k)
    # return the actual text for those indices
    return [chunks[i] for i in indices[0]]

from sentence_transformers import CrossEncoder  # for reranking

# load a cross‑encoder model that can score (query, doc) pairs
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, docs, top_n=5):
    # make pairs of (query, document) so the reranker can look at both
    pairs = [(query, doc) for doc in docs]
    # get a score for each pair from the reranker model
    scores = reranker.predict(pairs)
    # combine docs and scores and sort by score, highest first
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    # return only the top `top_n` documents
    return [doc for doc, score in ranked[:top_n]]

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# load the tokenizer and causal model we will use for generation
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    torch_dtype=torch.float16,
    device_map="auto"              # let HuggingFace place tensors on GPU/CPU
)

def generate_text(prompt, max_tokens=150):
    # turn the prompt into token ids and move them to the model’s device
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # ask the model to produce up to `max_tokens` more tokens
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.3)
    # turn the output tokens back into a string and remove special tokens
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def reformulate_query(original_query, context):
    # construct a simple prompt asking the LM to rewrite the query
    prompt = f"""
You are helping with multi-hop retrieval.

Original Question:
{original_query}

Context so far:
{context}

Generate a focused follow-up search query to retrieve missing information.
Return only the refined search query.
"""
    # use the same text generator to get the new query
    return generate_text(prompt)

def multi_hop_retrieval(query, max_hops=2):
    accumulated_context = []        # keep all passages found across hops
    current_query = query           # start with the question the user asked

    for hop in range(max_hops):
        print(f"Hop {hop+1} retrieval...")  # simple logging

        # get a batch of candidate passages
        retrieved = retrieve(current_query, top_k=15)
        # rerank them and keep the best ones
        reranked = rerank(current_query, retrieved, top_n=5)

        # add these best passages to our running context
        accumulated_context.extend(reranked)

        # if we are going to do another hop, refine the query first
        if hop < max_hops - 1:
            combined_context = "\n".join(reranked)
            current_query = reformulate_query(query, combined_context)

    # after all hops, give back everything we collected
    return accumulated_context

def build_final_prompt(query, contexts):
    # join the retrieved passages with blank lines between them
    context_text = "\n\n".join(contexts)
    # make the prompt that tells the LM to answer using those passages
    return f"""
You are a legal assistant.
Answer strictly using the provided documents.

Question:
{query}

Documents:
{context_text}

Answer:
"""

def multi_hop_rag(query):
    # perform the retrieval part
    contexts = multi_hop_retrieval(query, max_hops=2)
    # prepare the prompt with question + contexts
    final_prompt = build_final_prompt(query, contexts)
    # generate and return the final answer
    answer = generate_text(final_prompt, max_tokens=300)
    return answer

# try the system by asking a question about the constitution
response = multi_hop_rag(
    "Which case expanded Article 21 and what rights were added?"
)

# show the answer on the screen
print(response)