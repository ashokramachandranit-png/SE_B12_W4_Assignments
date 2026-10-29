import os
from crewai import Agent, Crew, LLM, Process, Task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

# 1. Load environment variables from local .env file
load_dotenv()

# 2. Set a custom local path for persistent memory storage (using raw string for Windows paths)
os.environ["CREWAI_STORAGE_DIR"] = os.path.abspath(r"D:\GenAI program\CrewAI\Crew_longterm_memory.db")

# 3. Initialize the web search tool
search_tool = SerperDevTool()

# 4. Configure robust OpenAI models
researcher_llm = LLM(
    model="openai/gpt-4o-mini", 
    temperature=0.1
)

writer_llm = LLM(
    model="openai/gpt-4o-mini", 
    temperature=0.5
)

# 5. Create Agents with explicit model assignments
researcher = Agent(
    role="Senior Tech Research Analyst",
    goal="Find absolute latest, real-time trending tech news stories published this month.",
    backstory=(
        "Expert tech journalist specializing in live web intelligence. "
        "You meticulously verify article publication timestamps to guarantee that "
        "all collected data matches the current month and year, completely filtering out legacy or outdated articles."
    ),
    tools=[search_tool],
    llm=researcher_llm,
    memory=True,
    verbose=True,
)

writer = Agent(
    role="Concise Newsletter Editor",
    goal="Synthesize raw research into a strict 5-line newsletter format.",
    backstory="Master of brief, high-impact editorial formatting.",
    tools=[],
    llm=writer_llm,
    verbose=True,
)

# 6. Define Tasks with explicit current-time constraints
research_task = Task(
    description=(
        "Search the web for the absolute latest top 5 tech news stories published during the current month (October 2026). "
        "Explicitly construct search queries targeting recent developments and ignore anything older than the current period. "
        "Extract headlines and key facts."
    ),
    expected_output="A structured summary of today's top 5 current tech news stories.",
    agent=researcher,
)

writer_task = Task(
    description="Take the current research findings and write a daily tech newsletter strictly following the 5-line structure.",
    expected_output=(
        "Line 1: [Top Story 1 Summary]\n"
        "Line 2: [Top Story 2 Summary]\n"
        "Line 3: [Top Story 3 Summary]\n"
        "Line 4: [Top Story 4 Summary]\n"
        "Line 5: [Top Story 5 Summary]\n"
        "Line 6: [Key Takeaway/Trend Analysis]\n"
        "Line 7: [Sign-off/Footer]"
    ),
    agent=writer,
    context=[research_task],
    output_file="tech_newsletter.md",
)

# 7. Assemble and Run the Crew with Memory & Embedder Settings matching 3072 dims
tech_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writer_task],
    process=Process.sequential,
    verbose=True,
    memory=True,
    embedder={
        "provider": "openai",
        "config": {"model_name": "text-embedding-3-large"}, # <--- Matches your existing 3072-dim store
    },
)

if __name__ == "__main__":
    print("## Running Crew Pipeline with GPT-4o & Live Search ##")
    result = tech_crew.kickoff()
    print("\n## Result ##")
    print(result)
