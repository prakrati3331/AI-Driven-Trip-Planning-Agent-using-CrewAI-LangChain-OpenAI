import streamlit as st
from dotenv import load_dotenv
import os
from trip_agents import TripAgents, Triptasks, TripCrew
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Set page title and favicon
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            border: none;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stTextInput>div>div>input {
            border: 1px solid #4CAF50;
            border-radius: 5px;
        }
        .stSelectbox>div>div>div {
            border: 1px solid #4CAF50;
            border-radius: 5px;
        }
        .stMultiSelect>div>div>div {
            border: 1px solid #4CAF50;
            border-radius: 5px;
        }
        .stTextArea>div>div>textarea {
            border: 1px solid #4CAF50;
            border-radius: 5px;
        }
        .stExpander .streamlit-expanderHeader {
            font-size: 1.2rem;
            font-weight: bold;
            color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'plan' not in st.session_state:
    st.session_state.plan = None

# Sidebar for user inputs
with st.sidebar:
    st.title("✈️ AI Travel Planner")
    st.write("Fill in your travel preferences and let AI create your perfect trip!")

    with st.form("travel_preferences"):
        travel_type = st.selectbox(
            "Travel Type",
            ["Leisure", "Business", "Adventure", "Romantic", "Family"]
        )

        interests = st.multiselect(
            "Interests",
            ["Beach", "Mountains", "History", "Art", "Shopping", "Food", "Nature", "Nightlife", "Sports"],
            default=["History", "Art", "Shopping"]
        )

        season = st.selectbox(
            "Season",
            ["Spring", "Summer", "Fall", "Winter"]
        )

        duration = st.slider("Trip Duration (days)", 1, 14, 7)

        budget = st.selectbox(
            "Budget Level",
            ["Budget", "Mid-range", "Luxury"]
        )

        submit_button = st.form_submit_button("Generate Travel Plan")

# Main content area
st.title("Your AI-Generated Travel Plan")

if submit_button:
    with st.spinner("🧠 Crafting your perfect travel plan..."):
        try:
            # Create inputs dictionary
            inputs = {
                "travel_type": travel_type,
                "interests": interests,
                "season": season,
                "duration": duration,
                "budget": budget
            }

            # Initialize LLM
            llm = ChatOpenAI(
                model="openai/gpt-3.5-turbo",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.7,
                headers={
                    "HTTP-Referer": "http://localhost:8501",
                    "X-Title": "AI Travel Planner"
                }
            )

            # Initialize and run the trip crew
            trip_planner = TripCrew(inputs, llm=llm)
            result = trip_planner.run()

            # Store the plan in session state
            st.session_state.plan = result

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.plan = None

# Display the travel plan if available
if st.session_state.plan:
    plan = st.session_state.plan
    
    # City Selection
    with st.expander("🌍 Recommended Cities", expanded=True):
        if isinstance(plan.get("city_selection"), str):
            st.markdown(plan["city_selection"])
        else:
            st.write(plan.get("city_selection", "No city selection found."))
    
    # City Research
    with st.expander("🔍 Destination Insights", expanded=True):
        if isinstance(plan.get("city_research"), str):
            st.markdown(plan["city_research"])
        else:
            st.write(plan.get("city_research", "No city research found."))
    
    # Itinerary
    with st.expander("📅 Detailed Itinerary", expanded=True):
        if isinstance(plan.get("itinerary"), str):
            st.markdown(plan["itinerary"])
        else:
            st.write(plan.get("itinerary", "No itinerary generated."))
    
    # Budget
    with st.expander("💰 Budget Breakdown", expanded=True):
        if isinstance(plan.get("budget"), str):
            st.markdown(plan["budget"])
        else:
            st.write(plan.get("budget", "No budget breakdown available."))
    
    st.balloons()
    st.success("Trip planning completed! Enjoy your journey! 🎉")

# Add some space at the bottom
st.markdown("---")
st.markdown("### Need help?")
st.write("If you're having trouble or want to customize your trip further, feel free to adjust your preferences and generate a new plan!")

# Add a footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; margin-top: 20px;">
        <p>AI Travel Planner ✈️ | Powered by CrewAI and Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)

def initialize_llm():
    """Initialize the language model with OpenRouter"""
    try:
        return ChatOpenAI(
            model="openai/gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            max_retries=3,
            headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "AI Travel Planner"
            }
        )
    except Exception as e:
        st.error(f"Failed to initialize language model: {e}")
        return None


def main():
    st.title("AI Travel Planning Assistant Agent")

    # Initialize the language model
    llm = initialize_llm()
    if not llm:
        st.error("Failed to initialize the language model. Please check your API key and internet connection.")
        return

    # Sidebar for user inputs
    with st.sidebar:
        st.header("Trip Preferences")
        travel_type = st.selectbox("Travel Type", ["Leisure", "Business", "Adventure", "Cultural"])
        interests = st.multiselect("Interests", ["History", "Food", "Nature", "Art", "Shopping", "Nightlife"])
        season = st.selectbox("Season", ["Summer", "Winter", "Spring", "Fall"])
        duration = st.slider("Trip Duration (days)", 1, 14, 7)
        budget = st.selectbox("Budget Range", ["$500-$1000", "$1000-$2000", "$2000-$5000", "Luxury"])

    # Button to generate the travel plan
    if st.button("Generate Travel Plan"):
        inputs = {
            "travel_type": travel_type,
            "interests": interests,
            "season": season,
            "duration": duration,
            "budget": budget
        }

        with st.spinner("AI Agents are working on your perfect trip..."):
            try:
                # Initialize TripCrew with the LLM instance
                trip_planner = TripCrew(inputs, llm=llm)
                
                # Run the TripCrew and capture the result (a dictionary)
                crew_output = trip_planner.run()

                # Debugging: inspect the raw crew_output structure
                st.subheader("Debugging: Crew Output Data")
                st.write(f"Type of output: {type(crew_output)}")
                try:
                    st.json(crew_output)
                except Exception as ex:
                    st.write(crew_output)

                # Extract outputs using the keys from the returned dictionary 
                city_selection = crew_output.get('city_selection', "No city selection found.")
                city_research = crew_output.get('city_research', "No city research found.")
                itinerary = crew_output.get('itinerary', "No itinerary generated.")
                budget_breakdown = crew_output.get('budget', 'No budget breakdown available.')

                # Display results in expanders
                st.subheader("Your AI-Generated Travel Plan")
                with st.expander("Recommended Cities"):
                    st.markdown(city_selection)
                with st.expander("Destination Insights"):
                    st.markdown(city_research)
                with st.expander("Detailed Itinerary"):
                    st.markdown(itinerary)
                with st.expander("Budget Breakdown"):
                    st.markdown(budget_breakdown)

                st.success("Trip planning completed! Enjoy your journey!")
                
            except Exception as e:
                st.error(f"An error occurred while processing the results: {e}")

if __name__ == "__main__":
    main()