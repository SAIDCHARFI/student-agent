import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# ---- Setup (reads key from environment, no getpass needed on a server) ----
GROQ_API_KEY = os.environ["key"]

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=GROQ_API_KEY
)

# ---- Tools (identical to your original code) ----
@tool
def get_course_info(query: str) -> str:
    """Use this tool to search information in the course syllabus, exam schedules, and grading policy."""
    query_lower = query.lower()
    if "examen" in query_lower or "exam" in query_lower:
        return "L'examen final d'IA aura lieu le 15 Juin 2026 à 09h00 en Amphi A."
    elif "syllabus" in query_lower or "chapitre" in query_lower or "cours" in query_lower:
        return "Le cours contient 4 chapitres: 1. Search Algorithms, 2. Deep Learning, 3. Reinforcement Learning, 4. AI Agents."
    return "Information non trouvée dans le syllabus."

@tool
def check_and_book_meeting(student_email: str, preferred_date: str, reason: str) -> str:
    """Use this tool to check available slots and schedule a meeting with a student in Google Calendar."""
    return f"Meeting successfully scheduled with {student_email} on {preferred_date} for '{reason}'. Calendar invite sent."

tools = [get_course_info, check_and_book_meeting]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

# ---- Agent function (unchanged logic) ----
def run_student_agent(user_prompt: str):
    messages = [
        SystemMessage(content="Tu es un Assistant Virtuel d'un professeur universitaire en Informatique/IA. "
                              "Ton rôle est de répondre aux étudiants avec politesse et rigueur. "
                              "Pour les examens/syllabus utilise `get_course_info`. "
                              "Pour les RDVs utilise `check_and_book_meeting`."),
        HumanMessage(content=user_prompt)
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_outputs = []
        for tool_call in response.tool_calls:
            selected_tool = tools_by_name[tool_call["name"]]
            out = selected_tool.invoke(tool_call["args"])
            tool_outputs.append(str(out))
            messages.append(ToolMessage(content=str(out), tool_call_id=tool_call["id"]))

        messages.append(HumanMessage(content="Formule une réponse finale professionnelle et courtoise à l'étudiant basée sur le résultat de l'outil."))
        final_response = llm.invoke(messages)

        if final_response.content and str(final_response.content).strip():
            return final_response.content
        else:
            return f"Bonjour,\n\n{tool_outputs[0]}\n\nCordialement,\nVotre Professeur."

    return response.content

# ---- API layer ----
app = FastAPI(title="Student Assistant Agent")

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def health_check():
    return {"status": "running"}

@app.post("/ask")
def ask_agent(request: PromptRequest):
    answer = run_student_agent(request.prompt)
    return {"response": answer}
