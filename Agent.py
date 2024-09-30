from langchain.agents.agent import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.base import BaseLanguageModel
from snowflake.connector import SnowflakeConnection

from toolkit import AgentToolkit


class Agent:
    agent_executor: AgentExecutor

    def __init__(self, con: SnowflakeConnection):
        toolkit = AgentToolkit(con=con)
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
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        # Snowflake Cortex replaces OpenAI for inference
        cortex_query = """
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            'mistral-7b',
            [
                {
                    'role': 'user',
                    'content': '{input}'
                }
            ],
            {}
        )
        """

        tools.insert(0, {"name": "Cortex", "tool": cortex_query})

        self.agent_executor = AgentExecutor(
            agent=None,  # Cortex-based agent replaces OpenAI agent
            tools=tools,
            verbose=True,
        )
    
    def get_executor(self) -> AgentExecutor:
        return self.agent_executor
