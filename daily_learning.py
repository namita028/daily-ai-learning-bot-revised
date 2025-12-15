import os
from dotenv import load_dotenv
load_dotenv()

import requests
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

LEARNING_CURRICULUM = [
    # Week 1-2: LLM Foundations
    "What is a Large Language Model (LLM) and how does it differ from traditional NLP",
    "Tokenization: How LLMs break text into pieces",
    "Word Embeddings: Converting words to numbers",
    "Attention Mechanism: How models focus on relevant parts",
    "Self-Attention and Query-Key-Value in Transformers",
    "The Transformer Architecture overview",
    "Pre-training vs Fine-tuning: Two phases of LLM training",
    "Next Token Prediction: How LLMs generate text",
    "Temperature and Sampling: Controlling randomness in outputs",
    "Context Window: Understanding input limits",

    # Week 3-4: Prompt Engineering
    "Zero-shot Prompting: Getting results without examples",
    "Few-shot Prompting: Teaching with examples",
    "Chain of Thought (CoT) Prompting",
    "System Prompts vs User Prompts",
    "Prompt Templates and Variables",
    "Output Formatting: JSON, Markdown, structured responses",
    "Prompt Injection and Security concerns",

    # Week 5-6: RAG (Retrieval Augmented Generation)
    "What is RAG and why do we need it",
    "Vector Databases: Storing and searching embeddings",
    "Chunking Strategies: How to split documents",
    "Semantic Search vs Keyword Search",
    "Embedding Models: Converting text to vectors",
    "Retrieval Pipeline: Query -> Search -> Context -> Generate",
    "Hybrid Search: Combining semantic and keyword search",
    "RAG Evaluation: Measuring retrieval quality",

    # Week 7-8: AI Agents
    "What is an AI Agent: Beyond simple chat",
    "Tool Use: Giving LLMs the ability to act",
    "Function Calling: Structured tool invocation",
    "ReAct Pattern: Reasoning and Acting",
    "Agent Memory: Short-term vs Long-term",
    "Multi-step Planning in Agents",
    "Agent Loops and Stopping Conditions",
    "Multi-Agent Systems: Agents working together",

    # Week 9-10: Advanced Concepts
    "Fine-tuning: When and how to customize models",
    "LoRA and Parameter-Efficient Fine-tuning",
    "RLHF: Reinforcement Learning from Human Feedback",
    "Evaluation Metrics for LLMs",
    "Hallucination: Why LLMs make things up",
    "Grounding: Keeping LLMs factual",
    "Cost Optimization: Tokens, caching, model selection",
    "Latency Optimization: Streaming, batching",

    # Week 11-12: Production & Ethics
    "LLM APIs: OpenAI, Anthropic, Groq, and others",
    "LangChain and LlamaIndex: Orchestration frameworks",
    "Guardrails: Content filtering and safety",
    "Observability: Logging and monitoring LLM apps",
    "A/B Testing LLM applications",
    "Bias in LLMs: Sources and mitigation",
    "Responsible AI: Ethics and best practices",
    "Future of LLMs: Multimodal, reasoning, and beyond",
]

PROMPT = """
You are an expert AI educator teaching a structured LLM learning curriculum.

Today's topic (Day {day_number} of {total_days}):
**{topic}**

Rules:
- Explain this specific topic thoroughly
- Assume the reader is a software engineer learning LLMs step by step
- Build on concepts from previous days (this is day {day_number})
- Avoid buzzwords and hype - be practical
- Include a concrete code example or mental model
- End with 1 reflective question or mini-exercise

Format STRICTLY as:

📌 **Day {day_number}: {topic}**

🧠 **Core Idea**
<Clear explanation of the concept>

⚙️ **Why It Matters**
<Practical importance in real systems>

💻 **Example**
<Code snippet or concrete example>

🔗 **Connection to Previous Concepts**
<How this builds on what was learned before>

❓ **Practice Exercise**
<A small exercise to reinforce learning>
"""

def get_day_number():
    """Calculate day number based on a start date (using UTC)."""
    from datetime import datetime, timezone
    start_date = datetime(2025, 12, 16, tzinfo=timezone.utc)  # Curriculum start date (UTC)
    today = datetime.now(timezone.utc)
    day_number = (today - start_date).days + 1
    # Loop back to day 1 after completing all topics
    if day_number > len(LEARNING_CURRICULUM):
        day_number = ((day_number - 1) % len(LEARNING_CURRICULUM)) + 1
    return max(1, day_number)

def generate_lesson():
    day_number = get_day_number()
    topic = LEARNING_CURRICULUM[day_number - 1]
    total_days = len(LEARNING_CURRICULUM)

    llm = ChatOpenAI(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
        temperature=0.7,
    )

    chain = (
        PromptTemplate.from_template(PROMPT)
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "day_number": day_number,
        "topic": topic,
        "total_days": total_days,
    })

def send_to_slack(text: str):
    response = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {os.environ['SLACK_BOT_TOKEN']}"},
        json={
            "channel": os.environ["SLACK_CHANNEL_ID"],
            "text": text,
        },
        timeout=10,
    )
    response.raise_for_status()

if __name__ == "__main__":
    day_number = get_day_number()
    topic = LEARNING_CURRICULUM[day_number - 1]
    print(f"Generating lesson for Day {day_number}: {topic}")

    lesson = generate_lesson()
    send_to_slack(lesson)
    print("Lesson sent to Slack successfully!")
