
import os
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process


# Load API key from .env in the same folder as this script
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)


# Country and familiar city options
COUNTRY_CITIES = {
    "India": [
        "Chennai",
        "Bengaluru",
        "Mumbai",
        "New Delhi",
        "Hyderabad",
        "Kochi",
        "Jaipur",
        "Kolkata",
        "Goa",
        "Agra",
    ],
    "United States": [
        "New York City",
        "Los Angeles",
        "San Francisco",
        "Chicago",
        "Las Vegas",
        "Miami",
        "Seattle",
        "Boston",
    ],
    "United Kingdom": [
        "London",
        "Manchester",
        "Liverpool",
        "Edinburgh",
        "Oxford",
        "Birmingham",
        "Bath",
    ],
    "France": [
        "Paris",
        "Nice",
        "Lyon",
        "Marseille",
        "Bordeaux",
        "Strasbourg",
    ],
    "Japan": [
        "Tokyo",
        "Kyoto",
        "Osaka",
        "Hiroshima",
        "Nara",
        "Sapporo",
    ],
    "Italy": [
        "Rome",
        "Venice",
        "Florence",
        "Milan",
        "Naples",
        "Pisa",
    ],
    "Australia": [
        "Sydney",
        "Melbourne",
        "Brisbane",
        "Perth",
        "Adelaide",
        "Canberra",
    ],
    "Singapore": [
        "Singapore",
    ],
}


def main():
    # Step 1: Verify the API key
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY was not found.")
        print(f"Please check your .env file: {env_path}")
        return

    # Step 2: Display country options
    countries = list(COUNTRY_CITIES.keys())

    print("\n========== CREWAI TRAVEL PLANNER ==========")
    print("\nSelect a country to explore:\n")

    for index, country in enumerate(countries, start=1):
        print(f"{index}. {country}")

    print(f"{len(countries) + 1}. Other country")

    country_choice = input(
        "\nEnter your choice: "
    ).strip()

    # Step 3: Get the selected country
    if not country_choice.isdigit():
        print("Invalid input. Please enter a number.")
        return

    choice = int(country_choice)

    if 1 <= choice <= len(countries):
        country = countries[choice - 1]
        cities = COUNTRY_CITIES[country]

    elif choice == len(countries) + 1:
        country = input(
            "Enter the country name: "
        ).strip()

        if not country:
            print("Country name cannot be empty.")
            return

        # Allow the user to enter any country
        cities = []

    else:
        print("Invalid country selection.")
        return

    # Step 4: Display familiar cities in the country
    if cities:
        print(f"\nPopular cities in {country}:\n")

        for index, city_name in enumerate(cities, start=1):
            print(f"{index}. {city_name}")

        print(f"{len(cities) + 1}. Other city")

        city_choice = input(
            "\nSelect a city: "
        ).strip()

        if not city_choice.isdigit():
            print("Invalid input. Please enter a number.")
            return

        city_index = int(city_choice)

        if 1 <= city_index <= len(cities):
            city = cities[city_index - 1]

        elif city_index == len(cities) + 1:
            city = input(
                f"Enter a city in {country}: "
            ).strip()

        else:
            print("Invalid city selection.")
            return

    else:
        city = input(
            f"Enter a city in {country}: "
        ).strip()

    if not city:
        print("City name cannot be empty.")
        return

    # Step 5: Create the Travel Planner Agent
    travel_planner = Agent(
        role="Travel Planner and Local Destination Expert",
        goal=(
            f"Recommend exactly three interesting places to visit "
            f"in {city}, {country}. Help travelers discover "
            "memorable attractions, local culture, and landmarks."
        ),
        backstory=(
            "You are an experienced travel planner who specializes "
            "in destinations around the world. You recommend "
            "interesting attractions, explain why each place is "
            "worth visiting, and provide practical travel advice. "
            "You aim to give clear, useful, and location-relevant "
            "recommendations to every traveler."
        ),
        verbose=True
    )

    # Step 6: Create one Task
    travel_task = Task(
        description=(
            f"Prepare a travel plan for {city}, {country}. "
            "Recommend exactly three distinct places to visit. "
            "For each place, provide its name, a short description, "
            "its location, and one reason to visit. "
            "Keep every recommendation relevant to the selected city. "
            "Do not invent attractions or claim that current opening "
            "hours have been verified."
        ),
        expected_output=(
            "A numbered list containing exactly 3 places to visit. "
            "For each place include: name, location, description, "
            "and reason to visit. End with one practical travel tip "
            "for the selected city."
        ),
        agent=travel_planner
    )

    # Step 7: Create the Crew
    travel_crew = Crew(
        agents=[travel_planner],
        tasks=[travel_task],
        process=Process.sequential,
        verbose=True
    )

    # Step 8: Execute the Crew
    print(f"\nPlanning your trip to {city}, {country}...\n")

    try:
        result = travel_crew.kickoff()

        print("\n========== YOUR TRAVEL PLAN ==========\n")
        print(result)

    except Exception as error:
        print("\nUnable to generate the travel plan.")
        print(f"Error details: {error}")


if __name__ == "__main__":
    main()
