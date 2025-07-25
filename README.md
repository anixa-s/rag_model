<h1> RAG Model for Efficient PDF Information Retrieval </h1>
This project uses a Retrieval-Augmented Generation (RAG) approach to extract and answer questions from long PDF documents, specifically targeted at helping students and educators quickly sift through dense International Baccalaureate (IB) material. Instead of spending hours skimming through IB study guides, past papers, or handbooks, this tool lets you ask natural-language questions and get instant, relevant answers—saving time and boosting productivity.

<h1> Problem Being Solved </h1>
<h3> The RAG model is built to help: </h3>
<ul> 
 <li>IB students quickly find key concepts, definitions, and explanations from large PDF files </li>
 <li>Educators and tutors retrieve accurate info from syllabi and IB guides </li>
 <li>Anyone working with long, academic PDFs that are too time-consuming to search manually</li>
</ul>

<h1> Features </h1>
<ul> 
 <li>Question-answering over PDFs using LLM + vector store </li>
 <li>Chunking of long documents for optimized search </li>
 <li>Fast semantic retrieval with embeddinGS</li>
 <li> Customizable for any subject or PDF</li>
</ul>

<h1> Setup Instructions (From Scratch) </h1>

## Clone the Repository 

```bash
git clone https://github.com/anixa-s/rag_model.git
cd rag_model
```

## Create and Activate a Virtual Environment

```bash
# On Mac/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Set Up API Keys
Create a .env file in the root directory and add your OpenAI API key:
```ini
OPENAI_API_KEY=your_api_key_here
```

## Query the database

Query the Chroma DB.

```python
python query_data.py "“How is the IB EE assessed?”
```
> You can get an API key from https://platform.openai.com/account/api-keys

