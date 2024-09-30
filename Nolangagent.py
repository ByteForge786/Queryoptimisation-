import json
from typing import List, Dict, Any, Callable
from snowflake.connector import SnowflakeConnection

class Tool:
    def __init__(self, name: str, func: Callable, description: str):
        self.name = name
        self.func = func
        self.description = description

    def run(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class Agent:
    def __init__(self, cortex_function: Callable, snowflake_connection: SnowflakeConnection):
        self.cortex_function = cortex_function
        self.snowflake_connection = snowflake_connection
        self.tools = self._get_tools()
        self.system_message = """
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

    def _get_tools(self) -> List[Tool]:
        return [
            Tool(
                name="snowflake_table_info",
                func=self._snowflake_table_info,
                description="Input: comma-separated list of tables. Output: schema and sample rows for those tables."
            ),
            Tool(
                name="query_checker",
                func=self._query_checker,
                description="Use this to check your query before executing it with query_executor."
            ),
            Tool(
                name="query_executor",
                func=self._query_executor,
                description="Input: correct SQL query. Output: result and query_id. If error, rewrite and try again."
            )
        ]

    def _snowflake_table_info(self, table_names: str) -> str:
        output_schema = ""
        _table_names = table_names.split(",")
        for t in _table_names:
            query = f"DESCRIBE TABLE {t}"
            cursor = self.snowflake_connection.cursor()
            result = cursor.execute(query).fetchall()
            cursor.close()
            output_schema += f"Schema for table {t}: {result}\n"
        return output_schema

    def _query_checker(self, query: str) -> str:
        template = """
        {query}
        Double check the Snowflake SQL query above for common mistakes, including:
        - Using NOT IN with NULL values
        - Using UNION when UNION ALL should have been used
        - Using BETWEEN for exclusive ranges
        - Data type mismatch in predicates
        - Properly quoting identifiers
        - Using the correct number of arguments for functions
        - Casting to the correct data type
        - Using the proper columns for joins

        If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

        Output the final SQL query only.

        SQL Query: """
        
        escaped_query = query.replace('"', '\\"').replace("'", "\\'")
        prompt = template.format(query=escaped_query)
        messages = [
            {
                'role': 'user',
                'content': prompt
            }
        ]
        return self.cortex_function(json.dumps(messages))

    def _query_executor(self, query: str) -> Dict[str, Any]:
        try:
            cursor = self.snowflake_connection.cursor()
            results = cursor.execute(query).fetchall()
            query_id = cursor.sfqid
            cursor.close()
            return {"results": results, "query_id": query_id}
        except Exception as e:
            return {"error": str(e)}

    def run(self, input_string: str) -> str:
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
        
        while True:
            response = self.cortex_function(json.dumps(messages))
            
            if response.strip().startswith("Final Answer:"):
                return response.split("Final Answer:")[-1].strip()
            
            tool_name = response.split(":")[0].strip()
            tool_input = ":".join(response.split(":")[1:]).strip()
            
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if tool:
                tool_response = tool.run(tool_input)
                messages.append({'role': 'user', 'content': f"Tool {tool_name} returned: {tool_response}"})
            else:
                messages.append({'role': 'user', 'content': f"Error: Tool {tool_name} not found."})

    def get_executor(self):
        return self.run
