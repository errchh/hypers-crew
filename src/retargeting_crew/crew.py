from crewai import Agent, Crew, Process, Task, LLM #, Knowledge
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool
##from crewai.knowledge import knowledge
#from base_knowledge_source import JSONKnowledgeSource
#from crewai.knowledge.source import BaseKnowledgeSource
#from crewai.knowledge.source import base_knowledge_source
##from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
##from crewai_tools import JSONSearchTool
import json 

# llm_temp_0 = LLM(
#     model="ollama/granite3.1-dense:8b",
#     temperature=0,
#     base_url="http://localhost:11434",
#     #api_key="your-api-key-here"
# )
llm = LLM(
    model="ollama/granite3.1-dense:8b",
    temperature=0.7,
    base_url="http://localhost:11434",
    #api_key="your-api-key-here"
)
# embedder = {
#     "provider": "ollama",
#     "config": {
#         "model": "granite-embedding:278m",
#         "base_url": "http://localhost:11434"
#     }
# }

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

# Instantiate tools
# file_reader = FileReadTool()
# file_read_tool_customer = FileReadTool('knowledge/customer_data.json')
# file_read_tool_catalogue = FileReadTool('knowledge/catalogue_data.json')
# file_read_tool_occassion = FileReadTool('knowledge/occassion_data.json')
# knowledge_source = base_knowledge_source()
# knowledge_source_cust = base_knowledge_source("customer_data.json")
# knowledge_source_cata = base_knowledge_source("catalogue_data.json")
# knowledge_source_occa = base_knowledge_source("occassion_data.json")

# Create a JSON knowledge source
# customer_source = JSONKnowledgeSource(
#     file_paths=["customer_data.json"]
# )
# catalogue_source = JSONKnowledgeSource(
#     file_paths=["catalogue_data.json"]
# )
# occassion_source = JSONKnowledgeSource(
#     file_paths=["occassion_data.json"]
# )

# For Knowledge object
# customer_knowledge = Knowledge(
#     collection_name="customer_knowledge",
#     sources=[customer_source],
#     embedder={
#         "provider": "ollama",
#         "config": {
#             "model": "snowflake-arctic-embed2:latest"  # or any other model you've pulled with Ollama
#         }
#     }
# )
# catalogue_knowledge = Knowledge(
#     collection_name="catalogue_knowledge",
#     sources=[catalogue_source],
#     embedder={
#         "provider": "ollama",
#         "config": {
#             "model": "snowflake-arctic-embed2:latest"  # or any other model you've pulled with Ollama
#         }
#     }
# )
# occassion_knowledge = Knowledge(
#     collection_name="occassion_knowledge",
#     sources=[occassion_source],
#     embedder={
#         "provider": "ollama",
#         "config": {
#             "model": "snowflake-arctic-embed2:latest"  # or any other model you've pulled with Ollama
#         }
#     }
# )

# Create knowledge with JSON source
# customer_knowledge = knowledge(
#     collection_name="customer_knowledge",
#     sources=[customer_source]
# )
# catalogue_knowledge = knowledge(
#     collection_name="catalogue_knowledge",
#     sources=[catalogue_source]
# )
# occassion_knowledge = knowledge(
#     collection_name="occassion_knowledge",
#     sources=[occassion_source]
# )

# Create the JSONSearchTool
#customer_json_tool = JSONSearchTool(json_path='knowledge/customer_data.json')
#catalogue_json_tool = JSONSearchTool(json_path='knowledge/catalogue_data.json')
#occassion_json_tool = JSONSearchTool(json_path='knowledge/occassion_data.json')

# Read JSON as python object 
customer_read_tool = FileReadTool(file_path='knowledge/customer_data.json')
# json_text = file_read_tool.run()
# customer_json_object = json.loads(json_text)

catalogue_read_tool = FileReadTool(file_path='knowledge/catalogue_data.json')
# json_text = file_read_tool.run()
# catalogue_json_object = json.loads(json_text)

occassion_read_tool = FileReadTool(file_path='knowledge/occassion_data.json')
# json_text = file_read_tool.run()
# occassion_json_object = json.loads(json_text)


# with open('knowledge/customer_data.json', 'r') as file:
#     json_text = file.read()
#     customer_data = json.loads(json_text)

# with open('knowledge/catalogue_data.json', 'r') as file:
#     json_text = file.read()
#     catalogue_data = json.loads(json_text)

# with open('knowledge/occassion_data.json', 'r') as file:
#     json_text = file.read()
#     occassion_data = json.loads(json_text)

@CrewBase
class RetargetingCrew():
	"""RetargetingCrew crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def customer_researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['customer_researcher'],
			#knowledge_sources=[customer_source],
			#embedder=embedder,
			tools=[customer_read_tool],
			verbose=True,
			llm=llm 
		)

	@agent
	def catalogue_researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['catalogue_researcher'],
			#knowledge_sources=[catalogue_source],
			#embedder=embedder,	
			tools=[catalogue_read_tool],
			verbose=True,
			llm=llm 
		)

	@agent
	def occassion_researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['occassion_researcher'],
			#knowledge_sources=[occassion_source],
			#embedder=embedder,
			tools=[occassion_read_tool],	
			verbose=True,
			llm=llm 
		)

	@agent
	def copywriter(self) -> Agent:
		return Agent(
			config=self.agents_config['copywriter'],
			verbose=True,
			llm=llm
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def research_customer(self) -> Task:
		return Task(
			config=self.tasks_config['research_customer'],
		)

	@task
	def research_catalogue(self) -> Task:
		return Task(
			config=self.tasks_config['research_catalogue'],
		)

	@task
	def research_occassion(self) -> Task:
		return Task(
			config=self.tasks_config['research_occassion'],
		)

	@task
	def write(self) -> Task:
		return Task(
			config=self.tasks_config['write'],
			output_file='retargeting_message.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the RetargetingCrew crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			memory=True,
			verbose=True
			#knowledge_sources=[customer_source, catalogue_source, occassion_source],
			#embedder=embedder				
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
