import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew

load_dotenv()

class TripAgents:
    def __init__(self):
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        
        # Configure the LLM with OpenRouter settings
        self.llm = ChatOpenAI(
            model="openai/gpt-3.5-turbo",
            openai_api_key=openai_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7,
            headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "AI Travel Planner"
            }
        )
    
    def city_selector_agent(self):
        return Agent(
            role='City Selection Expert',
            goal='Identify best cities to visit based on user preferences',
            backstory=(
                "An expert travel geographer with extensive knowledge about world cities "
                "and their cultural, historical, and entertainment offerings"
            ),
            llm=self.llm,
            verbose=True
        )
    
    def local_expert_agent(self):
        return Agent(
            role='Local Destination Expert',
            goal='Provide detailed insights about selected cities',
            backstory="A knowledgeable local guide with first-hand experience of the city's culture and attractions",
            llm=self.llm,
            verbose=True
        )
    
    def travel_planner_agent(self):
        return Agent(
            role='Professional Travel Planner',
            goal='Create detailed day-by-day itineraries',
            backstory="An experienced travel coordinator with perfect logistical planning skills",
            llm=self.llm,
            verbose=True
        )
    
    def budget_manager_agent(self):
        return Agent(
            role='Travel Budget Specialist',
            goal='Optimize travel plans to stay within budget',
            backstory="A financial planner specializing in travel budgets and cost optimization",
            llm=self.llm,
            verbose=True
        )

class Triptasks: 
    def __init__(self):
        pass
    
    def city_selection_task(self, agent, inputs):
        return Task(
        name="city_selection",
        description=(
            f"Analyze user preferences and select exactly 3 best destinations:\n"
            f"- Travel Type: {inputs['travel_type']}\n"
            f"- Interests: {', '.join(inputs['interests'])}\n"
            f"- Season: {inputs['season']}\n"
            f"- Budget: {inputs['budget']}\n\n"
            "IMPORTANT: Return ONLY a bullet-point list of exactly 3 cities with their countries, "
            "formatted like this:\n"
            "- Paris, France\n"
            "- Tokyo, Japan\n"
            "- New York, USA"
        ),
        agent=agent,
        expected_output="Bullet-point list of exactly 3 cities with countries"
    )

    def city_research_task(self, agent, city):
        return Task(
            name="city_research",
            description=(
                f"Provide detailed insights about {city} including:\n"
                f"- Top 5 attractions\n"
                f"- Local cuisine highlights\n"
                f"- Cultural norms/etiquette\n"
                f"- Recommended accommodation areas\n"
                f"- Transportation tips"
            ),
            agent=agent,
            expected_output="Organized sections with clear headings and bullet points."
        )

    def itinerary_creation_task(self, agent, inputs, city):
        return Task(
            name="itinerary",
            description=(
                f"Create a {inputs['duration']}-day itinerary for {city} including:\n"
                f"- Daily schedule with time allocations\n"
                f"- Activity sequencing\n"
                f"- Transportation between locations\n"
                f"- Meal planning suggestions"
            ),
            agent=agent,
            expected_output="Day-by-day table format with time slots and activity details."
        )

    def budget_planning_task(self, agent, inputs, itinerary):
        return Task(
            name="budget",
            description=(
                f"Create a budget plan for the selected budget range ({inputs['budget']}) covering:\n"
                f"- Accommodation costs\n"
                f"- Transportation expenses\n"
                f"- Activity fees\n"
                f"- Meal budget\n"
                f"- Emergency funds allocation"
            ),
            agent=agent,
            expected_output="Itemized budget table with total cost analysis."
        )

class TripCrew:
    def __init__(self, inputs, llm=None):
        self.inputs = inputs
        self.agents = TripAgents()
        self.tasks = Triptasks()
        if llm:
            self.agents.llm = llm

    def run(self):
        try:
            # Initialize agents
            city_selector = self.agents.city_selector_agent()
            local_expert = self.agents.local_expert_agent()
            travel_planner = self.agents.travel_planner_agent()
            budget_manager = self.agents.budget_manager_agent()

            # Create and run city selection task
            select_cities = self.tasks.city_selection_task(city_selector, self.inputs)
            city_crew = Crew(agents=[city_selector], tasks=[select_cities], verbose=True)
            city_selection_result = city_crew.kickoff()
            
            # Store city selection
            result = {
                "city_selection": city_selection_result,
                "city_research": "Please wait, researching selected city...",
                "itinerary": "Please wait, creating itinerary...",
                "budget": "Please wait, generating budget..."
            }
            
            # Extract the first city from the selection
            selected_city = self._extract_city_from_result(city_selection_result)
            if not selected_city:
                return {
                    "city_selection": city_selection_result,
                    "city_research": "Could not determine a city from the selection.",
                    "itinerary": "",
                    "budget": ""
                }
            
            # Create and run city research task
            research_task = self.tasks.city_research_task(local_expert, selected_city)
            research_crew = Crew(agents=[local_expert], tasks=[research_task], verbose=True)
            research_result = research_crew.kickoff()
            result["city_research"] = research_result
            
            # Create and run itinerary task
            itinerary_task = self.tasks.itinerary_creation_task(travel_planner, self.inputs, selected_city)
            itinerary_crew = Crew(agents=[travel_planner], tasks=[itinerary_task], verbose=True)
            itinerary_result = itinerary_crew.kickoff()
            result["itinerary"] = itinerary_result
            
            # Create and run budget task
            budget_task = self.tasks.budget_planning_task(budget_manager, self.inputs, itinerary_task)
            budget_crew = Crew(agents=[budget_manager], tasks=[budget_task], verbose=True)
            budget_result = budget_crew.kickoff()
            result["budget"] = budget_result
            
            return result
            
        except Exception as e:
            print(f"Error in run method: {str(e)}")
            return {
                "city_selection": f"Error: {str(e)}",
                "city_research": "",
                "itinerary": "",
                "budget": ""
            }

    def _extract_city_from_result(self, result):
        """Extract the first mentioned city from the city selection result."""
        print(f"Extracting city from result: {result}")
        
        if not result:
            print("No result provided")
            return None
            
        # If the result is a string, try to find a city name in it
        if isinstance(result, str):
            print("Result is a string, searching for city names...")
            # First, try to find a pattern like "- Paris, France" or "1. Paris, France"
            import re
            match = re.search(r'(?:^|\n|^[0-9]+[.)]\s*)([A-Z][a-zA-Z\s-]+?)(?:\s*,\s*[A-Z][a-zA-Z\s-]+)?(?:\n|$)', result)
            if match:
                city = match.group(1).strip()
                print(f"Found city: {city}")
                return city
            
            # If no match, try parsing as JSON
            import json
            try:
                parsed = json.loads(result)
                print(f"Successfully parsed as JSON: {parsed}")
                return self._extract_city_from_result(parsed)
            except json.JSONDecodeError:
                pass
        
        # If the result is a dictionary, try to get the city from it
        elif isinstance(result, dict):
            print("Result is a dictionary, searching for city keys...")
            # Check common keys that might contain the city
            for key in ['city', 'selected_city', 'city_name', 'name', 'output']:
                if key in result:
                    print(f"Found key '{key}' in result")
                    city = self._extract_city_from_result(result[key])
                    if city:
                        return city
        
        # If we get here, we couldn't find a city
        print("Could not extract city from result")
        return "Paris"  # Fallback to Paris if no city found

    def process_crew_results(self, result):
        """Process the raw output from the crew's execution."""
        print(f"Processing result of type: {type(result)}")
        
        # Initialize default result
        processed = {
            "city_selection": str(result) if result else "No result returned",
            "city_research": "Please complete city selection first.",
            "itinerary": "Please complete city research first.",
            "budget": "Please complete itinerary first."
        }


