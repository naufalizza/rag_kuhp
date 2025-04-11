import logging
import os, re
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
import json
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(filename='G:\\Shared\\proyek\\rag_exercise\\rag_kuhp\\log\\main.log',
                    filemode='w',
                    level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

def query_ollama(prompt, model="mistral:7b-instruct-q4_0"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False  # Set to True for streaming responses
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()

    return response.json()['response']

def load_json(json_path: str) -> list|dict:
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def load_pdf(pdf_path: str) -> list:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    for index, page in enumerate(pages):
        text = page.page_content
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('Biro Hukum dan Humas Badan Urusan Administrasi Mahkamah Agung-RI /g3/g3 ', '')
        text = text.replace('BUKU', '\nBUKU')
        text = text.replace('BAB', '\nBAB')
        text = text.replace('Pasal', '\nPasal')
        text = re.sub(r'\((\d+)\)', r'\n(\1)', text)
        text = re.sub(r'(\d+)\.', r'\n  \1.', text)
        text = re.sub(r'(\s\w+\s+)\n  (\d+)\.', r'\1\2.', text)
        text = re.sub(r'( [a-zA-Z])\.', r'\n\1.', text)
        text = re.sub(r'dan \n ( [a-zA-Z])\.', r'dan\1.', text)
        
        # case specific replacements
        text = text.replace(',dan 131 ', ',dan 131\n  ')
        text = text.replace('b. pidana tambahan ', 'b. pidana tambahan \n ')
        # text = text.replace('', '\n')
        pages[index].page_content = text
    return pages

def chunk_document(pages):
    related_kuhp = []
    current_book = None
    current_chapter = None
    current_article = None
    current_verse = None
    chunks = []

    for page_num, page in tqdm(enumerate(pages, 1), desc='chunking pages'):
        # if page_num > 5: break
        # logging.info(f"page-{page_num}")
        for line in page.page_content.split('\n'):
            trailing_l1 = None
            trailing_l2 = None
            if line.startswith('BUKU '):
                current_book = line
                current_chapter = None
                current_article = None
                current_verse = None
            elif line.startswith('BAB '):
                current_chapter = line
                current_article = None
                current_verse = None
            elif line.startswith('Pasal '):
                current_article = line
                current_verse = None
            elif bool(re.match(r'^\(d+\)', line)):
                current_verse = line
            if line.startswith('  '):
                trailing_l2 = line
            elif line.startswith(' '):
                trailing_l1 = line
            
            
            if (not current_book or not current_chapter or not current_article): continue
            related_line = ''
            related_line += f"\n{current_book}" if current_book else ''
            related_line += f"\n{current_chapter}" if current_chapter else ''
            related_line += f"\n{current_article}" if current_article else ''
            related_line += f"\n{current_verse}" if current_verse else ''
            if trailing_l1:
                if current_verse is not None:
                    current_verse += trailing_l1
                elif current_article is not None:
                    current_article += trailing_l1
            if trailing_l2 is not None:
                related_line += f"\n{trailing_l2}"
            else:
                related_line += f"\n{line}" if line != current_article else ''

            # logging.info(related_line.replace('\n', '. '))
            chunks.append(related_line)
    return chunks

def main():
    data_path = r'G:\Shared\proyek\rag_exercise\rag_kuhp\data\KUHP.pdf'
    queries_path = r'G:\Shared\proyek\rag_exercise\rag_kuhp\data\queries.json'

    pages = load_pdf(data_path)
    logger.info(f"num_pages: {len(pages)}")
    
    chunks = chunk_document(pages)
    chunks = [Document(page_content=chunk, metadata={'source': data_path}) for chunk in chunks]

    # model_name = "BAAI/bge-large-en-v1.5"
    model_name = "intfloat/multilingual-e5-base"
    model_kwargs = {'device':'cpu'}
    encode_kwargs = {'normalize_embeddings':True}

    logging.info("initializing embedding function...")
    embedding_function = HuggingFaceEmbeddings(
        model_name = model_name,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs
    )
    logging.info('done')

    logging.info("generating embeddings...")
    embeddings = []
    for chunk in tqdm(chunks, desc="embedding documents", total=len(chunks)):
        embedding = embedding_function.embed_query(chunk.page_content)
        embeddings.append((chunk.page_content, embedding))
    logging.info('done')

    logging.info("generating FAISS vector database...")
    faiss_db = FAISS.from_embeddings(embeddings, embedding_function)
    logging.info("done")

    # query = "hukuman mencuri apa saja?"
    
    for query_num, query in enumerate(load_json(queries_path), 1):
        prompt = f"Jawab dalam bahasa indonesia...\n{query}\nGunakan informasi dari dokumen-dokumen berikut dalam menjawab pertanyaan tersebut:\n"
        query_info = f"query #{query_num}: {query}"
        logging.info(query_info)
        matched_docs = faiss_db.similarity_search(query=query, k=5)
        for rank, doc in enumerate(matched_docs, 1):
            matched_doc_info = f"Dokumen #{rank}: {doc.page_content}"
            logging.info(matched_doc_info)
            prompt += f"\n{matched_doc_info}"
        print(prompt)
        logging.info(f"\nprompt:\n{prompt}")
        logging.info('generating answer using llm...')
        answer = query_ollama(prompt)
        logging.info('done')
        logging.info(f"\nanswer:\n{answer}")
        print(f"\nanswer:\n{answer}")
        break
        


    pass

if __name__ == '__main__':
    main()