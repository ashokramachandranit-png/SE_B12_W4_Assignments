
import os
from pathlib import Path

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM


# Load environment variables from the same folder as this script
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)


COUNTRY_CITIES = {
    "India": [
        "Chennai", "Bengaluru", "Mumbai", "New Delhi",
        "Hyderabad", "Kochi", "Jaipur", "Kolkata", "Goa", "Agra"
    ],
    "United States": [
        "New York City", "Los Angeles", "San Francisco",
        "Chicago", "Las Vegas", "Miami", "Seattle", "Boston"
    ],
    "United Kingdom": [
        "London", "Manchester", "Liverpool", "Edinburgh",
        "Oxford", "Birmingham", "Bath"
    ],
    "France": [
        "Paris", "Nice", "Lyon", "Marseille", "Bordeaux"
    ],
    "Japan": [
        "Tokyo", "Kyoto", "Osaka", "Hiroshima", "Nara", "Sapporo"
    ],
    "Italy": [
        "Rome", "Venice", "Florence", "Milan", "Naples", "Pisa"
    ],
    "Australia": [
        "Sydney", "Melbourne", "Brisbane", "Perth",
        "Adelaide", "Canberra"
    ],
    "Singapore": ["Singapore"]
}


def main():
    # Step 1: Read OpenAI configuration
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

    if not api_key:
        print("ERROR: OPENAI_API_KEY was not found in .env")
        print(f"Check the file at: {env_path}")
        return

    # Step 2: Configure the OpenAI LLM explicitly
    llm = LLM(
        model=f"openai/{model_name}",
        api_key=api_key,
        temperature=0.3
    )

    # Step 3: Ask the user to select a country
    countries = list(COUNTRY_CITIES.keys())

    print("\n========== CREWAI TRAVEL PLANNER ==========")
    print("\nSelect a country:\n")

    for index, country_name in enumerate(countries, start=1):
        print(f"{index}. {country_name}")

    print(f"{len(countries) + 1}. Other country")

    choice_text = input("\nEnter your choice: ").strip()

    if not choice_text.isdigit():
        print("Invalid input. Enter a number.")
        return

    choice = int(choice_text)

    if 1 <= choice <= len(countries):
        country = countries[choice - 1]
        cities = COUNTRY_CITIES[country]

    elif choice == len(countries) + 1:
        country = input("Enter the country name: ").strip()

        if not country:
            print("Country cannot be empty.")
            return

        cities = []

    else:
        print("Invalid country selection.")
        return

    # Step 4: Display familiar cities in the selected country
    if cities:
        print(f"\nPopular cities in {country}:\n")

        for index, city_name in enumerate(cities, start=1):
            print(f"{index}. {city_name}")

        print(f"{len(cities) + 1}. Other city")

        city_choice_text = input("\nSelect a city: ").strip()

        if not city_choice_text.isdigit():
            print("Invalid input. Enter a number.")
            return

        city_choice = int(city_choice_text)

        if 1 <= city_choice <= len(cities):
            city = cities[city_choice - 1]

        elif city_choice == len(cities) + 1:
            city = input(f"Enter a city in {country}: ").strip()

        else:
            print("Invalid city selection.")
            return
    else:
        city = input(f"Enter a city in {country}: ").strip()

    if not city:
        print("City cannot be empty.")
        return

    try:
        # Step 5: Create the Travel Planner Agent with OpenAI LLM
        travel_planner = Agent(
            role="Travel Planner and Local Destination Expert",
            goal=(
                f"Recommend exactly three interesting places to visit "
                f"in {city}, {country}, based on attractions, "
                "local culture, and visitor experience."
            ),
            backstory=(
                "You are an experienced travel planner who helps "
                "travelers discover memorable destinations. You explain "
                "why attractions are worth visiting and provide useful "
                "travel recommendations. You distinguish known facts "
                "from information that has not been verified."
            ),
            llm=llm,
            verbose=True
        )

        # Step 6: Create one Task
        travel_task = Task(
            description=(
                f"Recommend exactly three distinct places to visit "
                f"in {city}, {country}. For each place, provide its "
                "name, location, a short description, and one reason "
                "to visit. Keep every recommendation relevant to "
                "the selected city. Do not claim to have checked "
                "live opening hours or current conditions."
            ),
            expected_output=(
                "A numbered list containing exactly three places. "
                "For each place include its name, location, description, "
                "and reason to visit. End with one practical travel tip."
            ),
            agent=travel_planner
        )

        # Step 7: Assemble the Crew
        travel_crew = Crew(
            agents=[travel_planner],
            tasks=[travel_task],
            process=Process.sequential,
            verbose=True
        )

        # Step 8: Run the OpenAI-powered Crew
        print(f"\nDestination: {city}, {country}")
        print(f"LLM provider: OpenAI")
        print(f"Model: {model_name}")
        print("\nGenerating your travel recommendations...\n")

        result = travel_crew.kickoff()

        # Step 9: Display the result
        print("\n========== YOUR TRAVEL PLAN ==========\n")
        print(result)

    except Exception as error:
        print("\nUnable to generate the travel plan.")
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
