import os
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
        model="grok-2",
        api_key=os.environ["GROK_API_KEY"],
        temperature=0.7,
    )

    chain = (
        PromptTemplate.from_template(PROMPT)
        | llm
        | StrOutputParser()
    )

    return chain.invoke({})

def send_to_slack(text: str):
    payload = {"text": text}
    response = requests.post(
        os.environ["SLACK_WEBHOOK_URL"],
        json=payload,
        timeout=10,
    )
    response.raise_for_status()

if __name__ == "__main__":
    lesson = generate_lesson()
    send_to_slack(lesson)
