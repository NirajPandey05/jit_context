import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from jitcontext.index.embeddings import EmbedderSpec
from eval.locomo.runner import run_benchmark
from eval.locomo.scoring import LLMAnswerer, LLMJudge

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["GOOGLE_API_KEY"],
)

run_benchmark(
    path="data/locomo10.json",
    limit=10,
    embedder_spec=EmbedderSpec(
        provider="openai",
        model="models/gemini-embedding-001",
        client=client,
    ),
    answerer=LLMAnswerer(client, model="models/gemini-2.5-flash-lite", api="openai"),
    judge=LLMJudge(client,      model="models/gemini-2.5-flash-lite", api="openai"),
)
