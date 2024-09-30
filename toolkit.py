from typing import List, Optional, Type, Sequence, Dict, Any, Union, Tuple
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.tools import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
import json
from snowflake.connector import SnowflakeConnection

class _InfoSnowflakeTableToolInput(BaseModel):
    table_names: str = Field(
        ...,
        description=(
            "A comma-separated list of the table names for which to return the schema. "
            "Example input: 'table1, table2, table3'"
        ),
    )

class InfoSnowflakeTableTool(BaseTool):
    name: str = "snowflake_table_info"
    description: str = "Get the schema and sample rows for the specified Snowflake tables."
    args_schema: Type[BaseModel] = _InfoSnowflakeTableToolInput

    snowflake_connection: SnowflakeConnection = Field(exclude=True)

    def _run(
        self,
        table_names: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        output_schema = ""
        _table_names = table_names.split(",")
        for t in _table_names:
            query = f"DESCRIBE TABLE {t}"
            cursor = self.snowflake_connection.cursor()
            result = cursor.execute(query).fetchall()
            cursor.close()
            output_schema += f"Schema for table {t}: {result}\n"
        return output_schema

class _QueryCheckerToolInput(BaseModel):
    query: str = Field(..., description="A detailed SQL query to be checked.")

class QueryCheckerTool(BaseTool):
    name: str = "query_checker"
    description: str = """
    Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with query_executor!
    """
    args_schema: Type[BaseModel] = _QueryCheckerToolInput

    cortex_function: callable = Field(exclude=True)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
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

class _QueryExecutorToolInput(BaseModel):
    query: str = Field(..., description="A detailed and correct SQL query.")

class QueryExecutorTool(BaseTool):
    name: str = "query_executor"
    description: str = """
    Execute a SQL query against the Snowflake database and get back the result and query_id.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    args_schema: Type[BaseModel] = _QueryExecutorToolInput

    snowflake_connection: SnowflakeConnection = Field(exclude=True)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Union[str, Sequence[Dict[str, Any]]], Optional[str]]:
        try:
            cursor = self.snowflake_connection.cursor()
            results = cursor.execute(query).fetchall()
            query_id = cursor.sfqid
            cursor.close()
            return results, query_id
        except Exception as e:
            return f"Error: {e}", None

class AgentToolkit(BaseToolkit):
    snowflake_connection: SnowflakeConnection = Field(exclude=True)
    cortex_function: callable = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        info_snowflake_table_tool = InfoSnowflakeTableTool(
            snowflake_connection=self.snowflake_connection,
            description="Input: comma-separated list of tables. Output: schema and sample rows for those tables."
        )
        query_executor_tool = QueryExecutorTool(
            snowflake_connection=self.snowflake_connection,
            description="Input: correct SQL query. Output: result and query_id. If error, rewrite and try again."
        )
        query_checker_tool = QueryCheckerTool(
            cortex_function=self.cortex_function,
            description="Use this to check your query before executing it with query_executor."
        )
        return [
            query_executor_tool,
            info_snowflake_table_tool,
            query_checker_tool,
        ]
