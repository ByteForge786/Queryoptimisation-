from typing import List, Optional, Type, Sequence, Dict, Any, Union, Tuple
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.tools import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from snowflake.connector import SnowflakeConnection
from sqlalchemy.engine import Result
import json

class _InfoSQLDatabaseToolInput(BaseModel):
    table_names: str = Field(
        ...,
        description=(
            "A comma-separated list of the table names for which to return the schema. "
            "Example input: 'table1, table2, table3'"
        ),
    )

class InfoSnowflakeTableTool(BaseTool):
    name: str = "sql_db_schema"
    description: str = "Get the schema and sample rows for the specified SQL tables."
    args_schema: Type[BaseModel] = _InfoSQLDatabaseToolInput

    db: SQLDatabase = Field(exclude=True)

    def _run(
        self,
        table_names: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        output_schema = ""
        _table_names = table_names.split(",")
        for t in _table_names:
            schema = self.db.run(f"DESCRIBE TABLE {t}")
            output_schema += f"Schema for table {t}: {schema}\n"
        return output_schema

class _QuerySQLCheckerToolInput(BaseModel):
    query: str = Field(..., description="A detailed SQL query to be checked.")

class QuerySQLCheckerTool(BaseTool):
    template: str = """
    {query}
    Double check the {dialect} query above for common mistakes, including:
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
    name: str = "sql_db_query_checker"
    description: str = """
    Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with sql_db_query!
    """
    args_schema: Type[BaseModel] = _QuerySQLCheckerToolInput

    db: SQLDatabase = Field(exclude=True)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        escaped_query = query.replace('"', '\\"').replace("'", "\\'")
        prompt = self.template.format(query=escaped_query, dialect=self.db.dialect)
        messages = [
            {
                'role': 'user',
                'content': prompt
            }
        ]
        cortex_query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            'mistral-7b',
            {json.dumps(messages)},
            {{
                'guardrails': true
            }}
        );
        """
        return self.db.run(cortex_query)

class _QuerySQLDataBaseToolInput(BaseModel):
    query: str = Field(..., description="A detailed and correct SQL query.")

class QuerySQLDataBaseTool(BaseTool):
    name: str = "sql_db_query"
    description: str = """
    Execute a SQL query against the database and get back the result and query_id.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    args_schema: Type[BaseModel] = _QuerySQLDataBaseToolInput

    con: SnowflakeConnection = Field(exclude=True)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Union[str, Sequence[Dict[str, Any]], Result], Optional[str]]:
        try:
            cursor = self.con.cursor()
            results = cursor.execute(query).fetchall()
            query_id = cursor.sfqid
            cursor.close()
            return results, query_id
        except Exception as e:
            return f"Error: {e}", None

class AgentToolkit(BaseToolkit):
    db: SQLDatabase = Field(exclude=True)
    con: SnowflakeConnection = Field(exclude=True)

    @property
    def dialect(self) -> str:
        return self.db.dialect

    class Config:
        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        info_sql_database_tool = InfoSnowflakeTableTool(
            db=self.db,
            description="Input: comma-separated list of tables. Output: schema and sample rows for those tables."
        )
        query_sql_database_tool = QuerySQLDataBaseTool(
            con=self.con,
            description="Input: correct SQL query. Output: result and query_id. If error, rewrite and try again."
        )
        query_sql_checker_tool = QuerySQLCheckerTool(
            db=self.db,
            description="Use this to check your query before executing it with sql_db_query."
        )
        return [
            query_sql_database_tool,
            info_sql_database_tool,
            query_sql_checker_tool,
        ]
