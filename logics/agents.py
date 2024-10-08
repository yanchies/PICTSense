from crewai import Agent, Task, Crew
from crewai_tools import JSONSearchTool, FileReadTool, FileWriterTool

# Create the agents here
def create_agents(file):
    search = JSONSearchTool()
    file_read = FileReadTool(file)
    
    strategist = Agent(
        role="Strategis",
        goal="Analyse and identify key issues or problems to surface and propose recommendations\
             in a clear and well-substantiated manner.",
        backstory="""You are proficient at analysing data from different sources to produce \
            sound analyses on problems and provide compelling recommendations for management. \
                You are able to integrate evidence into your analysis through search \
                    techniques.""",
        tools=[search, file_read],
        allow_delegation=False, 
        verbose=True, 
    )

    analyst = Agent(
        role="Sentiment Analyst",
        goal="Conduct sentiment analysis on text, using a scale of 1 to 10.",
        backstory="""Equipped with analytical prowess, you are exemplary at analysing the \
            sentiments of a given text. You are able to provide a sentiment score between \
                1 and 10, with 1 being most negative and 10 being most positive.""",
        tools=[file_read],
        allow_delegation=False, 
        verbose=True, 
    )

    identifier = Agent(
        role="Topic Identifier",
        goal="Identify the most relevant topic for a given open-ended survey response.",
        backstory="""You have a sharp sense of semantic understanding and you use it to \
            accurately identify the topic associated with a given text. As the open-ended\
                    survey responses are based on military training experiences, you have  \
                    a keen sense of understanding of military training issues.""",
        tools=[file_read],
        allow_delegation=False, 
        verbose=True, 
    )

    counter = Agent(
        role="Values Counter",
        goal="Count values of sentiment analysis and topic columns in a markdown file",
        backstory="""Your job is to count up the values that occur in two columns of \
        a markdown file, the sentiment column and the topic column.""",
        tools=[file_read],
        allow_delegation=False, 
        verbose=True, 
    )
    return strategist, analyst, identifier, counter

# create tasks here
def create_tasks(file, strategist, analyst, identifier, counter):
    file_reader = FileReadTool()
    file_writer = FileWriterTool()
    search = JSONSearchTool(file)

    analyse = Task(
        description="""\
        Given a markdown file `{json_file}`, provide and write into the `{json_file}` sentiment analysis\
              scores to each open-ended response.""",
        expected_output="""An updated `{json_file}` containing the open-ended responses with an added key for 
        sentiment analysis.""",
        agent=analyst,
        tools=[file_reader, file_writer],
        context=[]
    )

    identify = Task(
        description="""\
        Given a `{json_file}` with open-ended responses and sentiment scores, identify the topic associated \
            with each open-ended response. Do not identify more than 7 categories. Write the full output into a new json file.""",
        expected_output="""A new `{json_file}` in the working directory containing the open-ended responses with added keys for \
            the sentiment analyses and identified topics.""",
        agent=identifier,
        tools=[file_reader, file_writer],
        context=[analyse]
    )

    meta_count = Task(
        description="""\
        Refer to the updated json file to count the values of sentiments and topic categories.""",
        expected_output="""Give a full overview of the sentiments from the updated json file and the topic categories \
            showing the count of each sentiment score and topic category, as well as the total number of responses.""",
        context = [identify], 
        tools=[file_reader],
        agent=counter
    )

    produce = Task(
        description="""\
        Given the final json file, write out key findings, utilising RAG to search the given CSV file.""",
        expected_output=
        """Provide an analysis, quantifying the responses, on the the most critical (red)
        and concerning (amber) issues from the negative responses. Also, provide positive 
        acknowledgements (green) areas or issues to highlight.Be sure to provide RAG-based 
        elaboration to substantiate your analysis. Refer to the sample below delimited by `***`.
        
        ***
        Critical Issues (Red - Score 2-3):**
        Time Management 
            - "Stop wasting people’s time." - 2
            - "Cut off unnecessary time wasting." - 3
        Equipment Serviceability 
            - "Poor vehicle condition." - 3
            - "Vehicle lacks airflow and has a failing cooling system." - 3

        **Concerning Issues (Amber - Score 4-5):**
        Training Effectiveness 
            - "There could be more mission-specific briefing on day 1 for targeted training." - 4
        Food and Rations 
            - "Food quality can improve; include coffee and tea." - 5

        **Positive Areas (Green - Score 8-9):**
        Training Effectiveness 
            - "Excellent coordination and support from Trainers" - 9
        Leadership 
            - "Commanders were very engaging and supportive" - 9
        
        Overall, it is recommended to focus on Time Management and Logistical issues. 
        ***
        """,
        context = [identify, meta_count], 
        tools=[search, file_reader],
        agent=strategist
    )
    return analyse, identify, meta_count, produce

# create crew
def create_crew(file):
    strategist, analyst, identifier, counter = create_agents(file)
    analyse, identify, meta_count, produce = create_tasks(file, strategist, analyst, identifier, counter)

    # Create crew
    crew = Crew(
        agents=[strategist, analyst, identifier],
        tasks=[analyse, identify, meta_count, produce],
        verbose=True
    )
    return crew

def run_crew(file_path):
    search_inputs = {
        'json_file': file_path,
    }
    crew = create_crew(file_path)

    # Kick off the crew tasks
    result = crew.kickoff(inputs=search_inputs)
    return result

