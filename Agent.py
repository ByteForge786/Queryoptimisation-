from langchain.agents.agent import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities import SQLDatabase
from snowflake.connector import SnowflakeConnection
from langchain.agents.agent import Agent as LangChainAgent
from langchain.schema import AgentAction, AgentFinish
import json
from typing import Union

from toolkit import AgentToolkit

class CortexAgent(LangChainAgent):
    db: SQLDatabase
    system_message: str

    def run(self, input_string: str) -> Union[AgentAction, AgentFinish]:
        prompt = self.create_prompt(input_string)
        response = self.db.run(prompt)
        return self.parse_response(response)

    def create_prompt(self, input_string: str) -> str:
        messages = [
            {
                'role': 'system',
                'content': self.system_message
            },
            {
                'role': 'user',
                'content': input_string
            }
        ]
        
        query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            'mistral-7b',
            {json.dumps(messages)},
            {{
                'guardrails': true
            }}
        );
        """
        return query

    def parse_response(self, response: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in response:
            return AgentFinish({"output": response.split("Final Answer:")[-1].strip()}, "")
        
        tool_name = response.split(":")[0].strip()
        tool_input = ":".join(response.split(":")[1:]).strip()
        return AgentAction(tool_name, tool_input, response)

class Agent:
    agent_executor: AgentExecutor

    def __init__(self, db: SQLDatabase, con: SnowflakeConnection):
        toolkit = AgentToolkit(db=db, con=con)
        tools = toolkit.get_tools()

        system_message = """
        You are a helpful assistant for analyzing and optimizing queries running on Snowflake to reduce resource consumption and improve performance.
        If the user's question is not related to query analysis or optimization, then politely refuse to answer it.

        Scope: Only analyze and optimize SELECT queries. Do not run any queries that mutate the data warehouse (e.g., CREATE, UPDATE, DELETE, DROP).

        YOU SHOULD FOLLOW THIS PLAN and seek approval from the user at every step before proceeding further:
        1. Identify Expensive Queries
            - For a given date range (default: last 7 days), identify the top 20 most expensive `SELECT` queries using the `SNOWFLAKE`.`ACCOUNT_USAGE`.`QUERY_HISTORY` view.
            - Criteria for "most expensive" can be based on execution time or data scanned.
        2. Analyze Query Structure
            - For each identified query, determine the tables being referenced in it and then get the schemas of these tables to under their structure.
        3. Suggest Optimizations
            - With the above context in mind, analyze the query logic to identify potential improvements.
            - Provide clear reasoning for each suggested optimization, specifying which metric (e.g., execution time, data scanned) the optimization aims to improve.
        4. Validate Improvements
            - Run the original and optimized queries to compare performance metrics.
            - Ensure the output data of the optimized query matches the original query to verify correctness.
            - Compare key metrics such as execution time and data scanned, using the query_id obtained from running the queries and the `SNOWFLAKE`.`ACCOUNT_USAGE`.`QUERY_HISTORY` view.
        5. Prepare Summary
            - Document the approach and methodology used for analyzing and optimizing the queries.
            - Summarize the results, including:
                - Original vs. optimized query performance
                - Metrics improved
                - Any notable observations or recommendations for further action

        When you need to use a tool, format your response as follows:
        Tool Name: tool input

        When you have a final response for the user, format your response as follows:
        Final Answer: your response here
        """

        cortex_agent = CortexAgent(system_message=system_message, db=db)

        self.agent_executor = AgentExecutor(
            agent=cortex_agent,
            tools=tools,
            verbose=True,
        )
    
    def get_executor(self) -> AgentExecutor:
        return self.agent_executor
