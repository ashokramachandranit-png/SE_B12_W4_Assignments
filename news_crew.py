import os
from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

# 1. Load environment variables from a local .env file
load_dotenv()

# 2. Set a custom local path for persistent memory storage (using a raw string)
os.environ["CREWAI_STORAGE_DIR"] = os.path.abspath(r"D:\GenAI program\CrewAI\Crew_longterm_memory.db")

# 3. Initialize tools & agents
search_tool = SerperDevTool()

researcher = Agent(
    role="Senior Tech Research Analyst",
    goal="Find today's top 3 trending tech news stories using web search.",
    backstory="Expert tech journalist tracking cutting-edge developments.",
    tools=[search_tool],
    memory=True,
    verbose=True,
)

writer = Agent(
    role="Concise Newsletter Editor",
    goal="Synthesize raw research into a well-spaced, clean newsletter format.",
    backstory="Master of brief, high-impact editorial formatting with clear spacing.",
    tools=[],
    verbose=True,
)

# 4. Define tasks
research_task = Task(
    description="Search the web for today's top 3 most important tech news stories.",
    expected_output="A structured summary of the top 3 tech stories.",
    agent=researcher,
)

writer_task = Task(
    description=(
        "Write a daily tech newsletter based exclusively on the research findings. "
        "Format the output clearly with distinct headings for each story (e.g., '### News 1: [Headline]'), "
        "followed by a concise summary description, and leave a blank line between each news item."
    ),
    expected_output=(
        "### News 1: [Headline]\n"
        "[Summary paragraph or details]\n\n"
        "### News 2: [Headline]\n"
        "[Summary paragraph or details]\n\n"
        "### News 3: [Headline]\n"
        "[Summary paragraph or details]"
    ),
    agent=writer,
    context=[research_task],
    output_file="tech_newsletter.md",
)

# 5. Assemble and Run the Crew with Memory Enabled
tech_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writer_task],
    process=Process.sequential,
    verbose=True,
    memory=True,  # Activates short-term, long-term, and entity memory
    embedder={
        "provider": "openai",
        "config": {"model_name": "text-embedding-3-small"},
    },
)

if __name__ == "__main__":
    print("## Running Crew Pipeline with Local Persistent Memory ##")
    result = tech_crew.kickoff()
    print("\n## Result ##")
    print(result)