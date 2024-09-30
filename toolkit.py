from typing import List, Optional, Type, Sequence, Dict, Any, Union, Tuple
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.tools import BaseToolkit
from snowflake.connector import SnowflakeConnection
from sqlalchemy.engine import Result
from your_snowflake_connector_file import Get_sf_data  # Importing your Snowflake connector

class _InfoSQLDatabaseToolInput(BaseModel):
    table_names: str = Field(
        ...,
        description=(
            "A comma-separated list of the table names for which to return the schema. "
            "Example input: 'table1, table2, table3'"
        ),
    )


class InfoSnowflakeTableTool(BaseTool):
    """Tool for getting metadata about a SQL database."""

    name: str = "sql_db_schema"
    description: str = "Get the schema and sample rows for the specified SQL tables."
    args_schema: Type[BaseModel] = _InfoSQLDatabaseToolInput

    def _run(
        self,
        table_names: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the schema for tables in a comma-separated list."""
        output_schema = ""
        _table_names = table_names.split(",")
        for t in _table_names:
            schema = Get_sf_data(f"DESCRIBE TABLE {t}")  # Use your Snowflake connector
            output_schema += f"Schema for table {t}: {schema}\n"
        return output_schema


class _QuerySQLDataBaseToolInput(BaseModel):
    query: str = Field(..., description="A detailed and correct SQL query.")


class QuerySQLDataBaseTool(BaseTool):
    """Tool for querying a SQL database."""

    name: str = "sql_db_query"
    description: str = """
    Execute a SQL query against the database and get back the result and query_id.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    args_schema: Type[BaseModel] = _QuerySQLDataBaseToolInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Union[str, Sequence[Dict[str, Any]], Result], Optional[str]]:
        """Execute the query, return the results and query_id; or an error message."""
        try:
            results = Get_sf_data(query)  # Use your Snowflake connector
            query_id = "1234"  # You may need to fetch the real query_id from the Snowflake connector if possible
            return results, query_id
        except Exception as e:
            return f"Error: {e}", None


class AgentToolkit(BaseToolkit):
    """Toolkit for interacting with SQL databases."""

    con: SnowflakeConnection = Field(exclude=True)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        info_sql_database_tool_description = (
            "Input to this tool is a comma-separated list of tables, output is the "
            "schema and sample rows for those tables. "
            "Example Input: table1, table2, table3"
        )
        info_sql_database_tool = InfoSnowflakeTableTool()

        query_sql_database_tool_description = (
            "Input to this tool is a detailed and correct SQL query, output is a "
            "result and query_id from the database."
        )
        query_sql_database_tool = QuerySQLDataBaseTool()

        return [info_sql_database_tool, query_sql_database_tool]

