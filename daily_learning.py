import os
from dotenv import load_dotenv
load_dotenv()

import requests
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

PROMPT = """
You are an expert AI educator and researcher.

Every day, generate ONE concise but deep learning lesson related to:
- Generative AI
- AI Agents
- Deep Learning
- Machine Learning

Rules:
- Focus on ONE concept only per day
- Assume the reader is a software engineer
- Avoid buzzwords and hype
- Explain why the concept matters in real systems
- Include a small example or mental model
- End with 1 reflective question or practical exercise

Format STRICTLY as:

📌 **Today's Concept**
<Concept name>

🧠 **Core Idea**
<Explanation>

⚙️ **Why It Matters**
<Why it matters>

🔍 **Example / Mental Model**
<Example>

❓ **Think About This**
<Question>
"""

def generate_lesson():
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

    return chain.invoke({})

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
    lesson = generate_lesson()
    send_to_slack(lesson)
