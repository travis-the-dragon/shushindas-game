import base64
import io
import json
import os
import random
import threading
import yaml
import numpy as np
import PyPDF2
from PIL import Image as PILImage

from google.cloud import aiplatform
from google.cloud import bigquery
import vertexai

from vertexai.language_models import TextGenerationModel, TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, Part, FinishReason, Candidate
from openai import OpenAI

import chromadb
from pydantic import BaseModel, Field
from typing import Union
import re
import weave

# Helper function that reads from the config file.
def get_config_value(config, section, key, default=None):
    """
    Retrieve a configuration value from a section with an optional default value.
    
    Args:
        config (dict): The loaded configuration from a YAML file.
        section (str): The section in the configuration to search.
        key (str): The key to retrieve the value for.
        default (Optional): The default value to return if the key is not found.
        
    Returns:
        The value from the configuration or the default if not found.
    """
    try:
        return config[section][key]
    except:
        return default

# Open the config file (config.yaml)
with open('./config.yaml') as f:
    config = yaml.safe_load(f)

# Read application variables from the config file
TITLE = get_config_value(config, 'app', 'title', 'The Desk of Shushinda')
SUBTITLE = get_config_value(config, 'app', 'subtitle', 'Beware of librarian, she\'s a real cat-astrophy!')
CONTEXT = get_config_value(config, 'palm', 'context',
                           'You are Shushinda Hushwisper, the infamous fictional cat wizard who lives in Discworld.')
BOTNAME = get_config_value(config, 'palm', 'botname', 'Shushinda')
TEMPERATURE = get_config_value(config, 'palm', 'temperature', 0.8)
MAX_OUTPUT_TOKENS = get_config_value(config, 'palm', 'max_output_tokens', 256)
TOP_P = get_config_value(config, 'palm', 'top_p', 0.8)
TOP_K = get_config_value(config, 'palm', 'top_k', 40)

PREAMBLE = get_config_value(config, 'shushinda', 'preamble')
MY_BIO = get_config_value(config, 'shushinda', 'my_bio')
PERSONALITY_TRAITS = get_config_value(config, 'shushinda', 'personality_traits')
SOME_FACTS = get_config_value(config, 'shushinda', 'some_facts')

SHUSHINDA = PREAMBLE + MY_BIO + PERSONALITY_TRAITS + SOME_FACTS

GREETINGS = get_config_value(config, 'shushinda', 'greetings')

ALL_SAMPLE_QUESTIONS = get_config_value(config, 'shushinda', 'sample_questions')

COULD_NOT_ANSWER = get_config_value(config, 'shushinda', 'say_what')

UNLOCK_QUESTION = "What is the Trial?"

LLMS = [
    {"model": "gpt-4o-mini", "family": "openai"},
    {"model": "gpt-4o", "family": "openai"},
    {"model": "gpt-4-turbo", "family": "openai"},
]

class SystemPrompt(weave.Object):
    """Class to define a system prompt in Weave."""
    prompt: str

class LanguageModel(weave.Model):
    """Class representing a Language Model with various configurations and operations."""

    name: str = Field("gpt-4o-mini", description="The name of the model. Used for nice display in Weave UI.")
    llm_fam: str = Field("openai", description="The family of the language model")
    llm_model_name: str = Field("gpt-4o-mini", description="The specific model name of the LLM")
    system_prompt: str = Field("Shushinda Hushwisper", description="The system prompt used by the LLM")
    # txt_model: Union[GenerativeModel, OpenAI] = Field(None, description="The model client to then .predict")

    def __init__(self, name: str, llm_name: str, prompt: str = SHUSHINDA):
        """Initialize the LanguageModel with configurations."""
        super().__init__()

        # Find the LLM configuration based on the provided llm_name
        llm = next(llm for llm in LLMS if llm["model"] == llm_name)

        print(f"Model: {llm['family']}")
        self.llm_fam = llm["family"]
        self.llm_model_name = llm["model"]
        self.name = name

        self.system_prompt = SystemPrompt(prompt=prompt)
        weave.publish(self.system_prompt)

    @weave.op
    def predict(self, question: str, image: PILImage.Image = None, context: str = None):
        """Predict the response based on the input question, optional image, and context.

        Args:
            question (str): The input question.
            image (PIL.Image, optional): An image pasted by the user (logged by Weave).
            context (str, optional): Additional context for the model.

        Returns:
            dict: Response containing the model's output and call ID.
        """
        resp: str = ""
        current_call = weave.get_current_call()

        if context is None:
            emb_stuff = EmbeddingsDB()
            # Get context data to answer the question based on embeddings
            context = emb_stuff.search_vector_database(question)

        if self.llm_fam == "gemini":
            resp = self.ask_gemini(question, context, image=image)
        elif self.llm_fam == "openai":
            resp = self.ask_openai(question, context, image=image)
        return {"response": resp, "call_id": current_call.id}

    @weave.op
    def ask_openai(self, question: str, data: str, image: PILImage.Image = None):
        """Request a response from the OpenAI model.

        Args:
            question (str): The input question.
            data (str): Additional data or context.
            image (PIL.Image, optional): An image to include in the request.

        Returns:
            str: The model's response or a fallback if the response is incomplete.
        """
        client = OpenAI()

        if image is not None:
            buf = io.BytesIO()
            fmt = image.format or 'PNG'
            image.save(buf, format=fmt)
            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            data_url = f"data:image/{fmt.lower()};base64,{b64}"
            user_content = [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
        else:
            user_content = question

        messages = [
            {"role": "system", "content": self.system_prompt.prompt},
            {"role": "user", "content": user_content}
        ]

        if data is not None:
            messages.extend([{"role": "assistant", "content": data}])

        response = client.chat.completions.create(
            model=self.llm_model_name,
            messages=messages
        )

        if response.choices[0].finish_reason != "stop":
            return random.choice(COULD_NOT_ANSWER)
        else:
            return response.choices[0].message.content

    @weave.op
    def ask_gemini(self, question: str, data: str, image: PILImage.Image = None):
        """Request a response from the Gemini model.

        Args:
            question (str): The input question.
            data (str): Additional data or context.
            image (PIL.Image, optional): An image to include in the request.

        Returns:
            str: The model's response or a fallback if the response is incomplete.
        """
        PROJECT_ID = os.getenv("PROJECT")
        REGION = os.getenv("REGION")
        vertexai.init(project=PROJECT_ID, location=REGION)
        txt_model = GenerativeModel(self.llm_model_name)

        PROMPT = f"""
        {self.system_prompt.prompt}
        CONTEXT: {data}
        QUESTION: {question}
        """

        parts = [PROMPT]
        if image is not None:
            buf = io.BytesIO()
            fmt = image.format or 'PNG'
            image.save(buf, format=fmt)
            parts.append(Part.from_data(data=buf.getvalue(), mime_type=f"image/{fmt.lower()}"))

        response = txt_model.generate_content(
            parts,
            stream=False,
        )

        if response.candidates[0].finish_reason != FinishReason.STOP:
            return random.choice(COULD_NOT_ANSWER)
        else:
            return response.text
class EmbeddingsDB:
    """Class to manage embedding models and vector databases for text processing."""

    def __init__(self, emb_model_fam="openai", vector_db="chroma"):
        """Initialize the EmbeddingsDB with model family and vector database configurations.

        Args:
            emb_model_fam (str): The embedding model family to use ("openai" or "gecko").
            vector_db (str): The vector database to use ("chroma" or "bigquery").
        """
        self.emb_model = None
        self.model_fam = emb_model_fam
        self.dimensions = 768
        self.vector_db = vector_db

        if emb_model_fam == "gecko":
            self.emb_model_name = "text-embedding-004"
            self.emb_model = TextEmbeddingModel.from_pretrained(self.emb_model_name)
        elif emb_model_fam == "openai":
            self.emb_model_name = "text-embedding-3-small"
            client = OpenAI()
            self.emb_model = client.embeddings
            self.dimensions = 1536

        if vector_db == "chroma":
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="shushinda-index",
                metadata={"hnsw:space": "cosine"},
            )
        elif vector_db == "bigquery":
            self.bq_client = bigquery.Client()

        print(f"Emb Model: {emb_model_fam}")

    def get_text(self, the_filename=None, the_text=None):
        """Returns an array of text from a file or directly provided text.

        Args:
            the_filename (str, optional): The filename and path of the file to ingest.
            the_text (str, optional): The actual text to ingest.

        Returns:
            list[str]: An array of text.
        """
        all_text = []

        if the_text is not None:
            all_text = [the_text]
            return all_text

        if the_filename and the_filename.endswith(('.pdf', '.PDF')):
            with open(the_filename, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                num_pages = len(pdf_reader.pages)

                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    all_text.append(page_text)
            return all_text

        elif the_filename and any(the_filename.endswith(ext) for ext in ['.txt', '.TXT', '.html', '.htm']):
            with open(the_filename, 'r') as txt_file:
                all_text = [txt_file.read()]
            return all_text

        return all_text

    def get_chunks(self, text):
        """Returns an array of chunks from a large body of text.

        Args:
            text (list[str]): A large body of text.

        Returns:
            list[str]: An array of chunks.
        """
        sentences = re.split(r'(?<=[.!?])\s+', "".join(text))
        return [s.strip() for s in sentences if s.strip()]

    def get_embeddings(self, chunk):
        """Returns an array of embedding vectors using the selected embedding model.

        Args:
            chunk (str): A chunk of text to generate embeddings for.

        Returns:
            list: An array of embedding vectors.
        """
        print('      Getting embeddings...')
        embs = []

        # time.sleep(1)  # Uncomment if rate limiting is needed
        if self.model_fam == "gecko":
            result = self.emb_model.get_embeddings([chunk])
            embs = result[0].values
        elif self.model_fam == "openai":
            chunk = chunk.replace("\n", " ")
            embs = self.emb_model.create(input=[chunk], model=self.emb_model_name).data[0].embedding

        return embs

    @weave.op
    def search_vector_database(self, question):
        """Search the vector database for the closest embeddings to the user's question.

        Args:
            question (str): The user's question.

        Returns:
            str: The concatenated documents that match the query.
        """
        the_embeddings = self.get_embeddings(question)

        if self.vector_db == "bq":
            data = self.search_bq(the_embeddings)
        elif self.vector_db == "chroma":
            data = self.search_chroma(the_embeddings)

        return data

    @weave.op
    def search_bq(self, the_embeddings):
        """Searches BigQuery for the closest matching documents.

        Args:
            the_embeddings (list): The embeddings of the query.

        Returns:
            str: Concatenated results from the search.
        """
        query = f"""
        SELECT base.id, base.doc_name, base.chunk_text, distance 
        FROM VECTOR_SEARCH(
            TABLE sushindas_stuff.chunks, 
            'chunk_vector',
            (select @search_embedding as embedding),
            'embedding',
            top_k => 2,
            distance_type => 'EUCLIDEAN' -- change to COSINE or EUCLIDEAN
        )
        ORDER BY distance ASC;
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("search_embedding", "FLOAT", the_embeddings),
            ]
        )
        query_job = self.bq_client.query(query, job_config=job_config)  # Make an API request.

        data = ""
        for row in query_job:
            data += row.chunk_text
            # print(f"id: {row.id}, doc_name: {row.doc_name}, chunk_text: {row.chunk_text}, distance: {row.distance}")

        return data

    @weave.op
    def search_chroma(self, the_embeddings):
        """Searches ChromaDB for the closest matching documents.

        Args:
            the_embeddings (list): The embeddings of the query.

        Returns:
            str: Concatenated chunk texts from the top 3 results.
        """
        results = self.chroma_collection.query(
            query_embeddings=[the_embeddings],
            n_results=3,
            include=["metadatas"],
        )
        top_texts = [m["chunk_text"] for m in results["metadatas"][0] if "chunk_text" in m]
        return "".join(top_texts)

    def insert_recs(self, rows_to_insert):
        """Insert records into the vector database based on the current configuration.

        Args:
            rows_to_insert (list[dict]): The records to insert.
        """
        if self.vector_db == "chroma":
            self.insert_chroma(rows_to_insert)
        elif self.vector_db == "bigquery":
            self.insert_bq(rows_to_insert)

    def insert_chroma(self, rows_to_insert):
        """Insert records into the ChromaDB collection.

        Args:
            rows_to_insert (list[dict]): The records to insert.
        """
        self.chroma_collection.upsert(
            ids=[str(r["id"]) for r in rows_to_insert],
            embeddings=[r["chunk_vector"] for r in rows_to_insert],
            metadatas=[
                {"doc_name": r["doc_name"], "chunk_text": r["chunk_text"], "chunk_id": str(r["chunk_id"])}
                for r in rows_to_insert
            ],
        )
        print(f" Upserted {len(rows_to_insert)} records to ChromaDB ")

    def insert_bq(self, rows_to_insert):
        """Inserts rows into the given BigQuery table.

        Args:
            rows_to_insert (list[dict]): The rows to insert.

        Returns:
            list: Errors encountered while inserting rows, if any.
        """
        batch_size = 100
        dataset = "shushindas_stuff"
        table_id = f"{dataset}.docs"

        errors = []
        for i in range(0, len(rows_to_insert), batch_size):
            batch = rows_to_insert[i:i + batch_size]
            errors = self.bq_client.insert_rows_json(table_id, batch)

        if not errors:
            print("New rows have been added.")
        else:
            print("Encountered errors while inserting rows: {}".format(errors))

        return errors

    def get_doc_names(self):
        """Retrieve unique document names from the ChromaDB collection.

        Returns:
            list: A sorted list of unique document names.
        """
        result = self.chroma_collection.get(include=["metadatas"])
        unique_doc_names = {m["doc_name"] for m in result["metadatas"] if "doc_name" in m}
        return sorted(unique_doc_names)


# ---------------------------------------------------------------------------
# Shared ChromaDB instance — PersistentClient crashes if opened concurrently
# from multiple threads, so we reuse a single EmbeddingsDB across the process.
# ---------------------------------------------------------------------------

_db_lock = threading.Lock()
_shared_db: "EmbeddingsDB | None" = None


def _get_db() -> "EmbeddingsDB":
    global _shared_db
    if _shared_db is None:
        with _db_lock:
            if _shared_db is None:
                _shared_db = EmbeddingsDB()
    return _shared_db


# ---------------------------------------------------------------------------
# Agent tools — standalone @weave.op functions so each call gets its own trace
# ---------------------------------------------------------------------------

OVERDUE_BOOKS = [
    ("Wand Maintenance for the Perpetually Clumsy", "Barnaby Fizzwick", 47),
    ("A Practical Guide to Ignoring Rules", "Mirabel Thorne", 312),
    ("Taming Grumpy Enchantments", "Professor Cedric Snoot", 8),
    ("1001 Things to Do With a Cauldron (Besides Soup)", "Heloise Dusk", 93),
    ("The Whispering Catalogue of Forbidden Snacks", "Anonymous", 201),
    ("Dragon Etiquette for the Socially Anxious", "Griselda Puff", 17),
    ("Advanced Napping in Restricted Sections", "Slumberwick H. Moss", 556),
    ("Why Your Familiar Is Judging You", "Ophelia Brank", 29),
]

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_library",
            "description": "Search the library stacks for information relevant to a topic. Returns matching text from ingested documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The topic or question to search for"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_books",
            "description": "List all documents and tomes available in the library. Call this first to understand what sources exist before searching.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_random_fact",
            "description": "Retrieve a random interesting fact or passage from the library's collection. Useful when the patron wants to be surprised or entertained.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_overdue_books",
            "description": "Check the overdue books ledger to see which tomes have been borrowed and not returned. Returns a notice about currently overdue items.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


@weave.op
def search_library(query: str) -> str:
    """Search ChromaDB for chunks relevant to the query."""
    return _get_db().search_vector_database(query)


@weave.op
def list_books() -> str:
    """Return a comma-separated list of all ingested document names."""
    books = _get_db().get_doc_names()
    return ", ".join(books) if books else "No books have been ingested yet."


@weave.op
def get_random_fact() -> str:
    """Return a random substantive passage from the library's collection."""
    db = _get_db()
    result = db.chroma_collection.get(include=["metadatas"])
    metadatas = result.get("metadatas", [])
    if not metadatas:
        return "The library shelves are bare — nothing has been ingested yet."
    # Exclude TOC/index chunks: too short or heavy with dot leaders
    candidates = [
        m for m in metadatas
        if len(m.get("chunk_text", "")) >= 200 and "....." not in m.get("chunk_text", "")
    ]
    if not candidates:
        candidates = metadatas
    chosen = random.choice(candidates)
    return f"From '{chosen.get('doc_name', 'unknown tome')}': {chosen.get('chunk_text', '')}"


@weave.op
def check_overdue_books() -> str:
    """Return a notice about overdue books from the ledger."""
    count = random.randint(1, 3)
    selection = random.sample(OVERDUE_BOOKS, min(count, len(OVERDUE_BOOKS)))
    lines = ["Overdue notices from Shushinda's ledger:"]
    for title, borrower, days in selection:
        lines.append(f"  • \"{title}\" by {borrower} — {days} days overdue")
    lines.append("Fines are payable in enchanted tokens or particularly good biscuits.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ShushindaAgent — replaces LanguageModel for the /ask endpoint
# ---------------------------------------------------------------------------

class ShushindaAgent(weave.Model):
    """Agent that uses OpenAI function calling to search the library before answering."""

    name: str = Field(default="ShushindaAgent")
    llm_model_name: str = Field(default="gpt-4o-mini")
    llm_fam: str = Field(default="openai")
    system_prompt: str = Field(default=SHUSHINDA)
    temperature: float = Field(default=0.7)

    def __init__(self, llm_model_name: str = "gpt-4o-mini", **kwargs):
        llm_config = next(
            (l for l in LLMS if l["model"] == llm_model_name),
            {"model": llm_model_name, "family": "openai"},
        )
        super().__init__(
            name=llm_model_name,
            llm_model_name=llm_model_name,
            llm_fam=llm_config["family"],
            **kwargs,
        )

    @weave.op
    def predict(self, question: str, image: PILImage.Image = None):
        current_call = weave.get_current_call()

        client = OpenAI()

        if image is not None:
            buf = io.BytesIO()
            fmt = image.format or "PNG"
            image.save(buf, format=fmt)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            mime = f"image/{fmt.lower()}"
            user_content = [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]
        else:
            user_content = question

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        tool_steps = []

        while True:
            response = client.chat.completions.create(
                model=self.llm_model_name,
                temperature=self.temperature,
                messages=messages,
                tools=AGENT_TOOLS,
                tool_choice="auto",
            )
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls":
                messages.append(choice.message)
                for tc in choice.message.tool_calls:
                    args = json.loads(tc.function.arguments)
                    if tc.function.name == "search_library":
                        result = search_library(args["query"])
                    elif tc.function.name == "list_books":
                        result = list_books()
                    elif tc.function.name == "get_random_fact":
                        result = get_random_fact()
                    elif tc.function.name == "check_overdue_books":
                        result = check_overdue_books()
                    else:
                        result = "Unknown tool."
                    tool_steps.append({"tool": tc.function.name, "args": args})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    })
            else:
                return {
                    "response": choice.message.content or random.choice(COULD_NOT_ANSWER),
                    "call_id": current_call.id,
                    "tool_steps": tool_steps,
                }