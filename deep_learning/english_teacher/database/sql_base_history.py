import pyodbc
import os
import logging

logger = logging.getLogger(__name__)


class InsertDatabase:
    def __init__(self):
        self.server = os.environ.get('server_database')
        self.table = 'EnglishConversation'
        self.database = 'master'
        self.trusted_connection = 'yes'
        self.connection_string = (f'DRIVER={{SQL Server}};SERVER={self.server};DATABASE={self.database};'
                                  f'Trusted_Connection={self.trusted_connection};')

    def insert_history_data(self, user_categoric, text_information, date_time=None):
        if not self.server:
            logger.info("Error: The environment variable 'server_database' was not found.")
            return

        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()

                query = """
                    INSERT INTO EnglishConversation
                        (user_categoric, text_information, date_time)
                    VALUES (?, ?, COALESCE(?, SYSDATETIME()))"""

                cursor.execute(query, (user_categoric, text_information, date_time))

            logger.info("Insert successfully completed.")
            return True

        except Exception as e:
            logger.debug(f"An error occurred: {str(e)}")
            return False

    def getting_data(self):
        if not self.server:
            logger.info("Error: The environment variable 'server_database' was not found.")
            return []

        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()

                query = """
                    WITH RecentMessages AS (
                        SELECT TOP 10
                            Id,
                            user_categoric,
                            text_information,
                            date_time
                        FROM EnglishConversation
                        ORDER BY date_time DESC, Id DESC
                    )
                    SELECT
                        user_categoric,
                        text_information,
                        date_time
                    FROM RecentMessages
                    ORDER BY date_time ASC, Id ASC;
                """

                cursor.execute(query)
                data_history = cursor.fetchall()
                logger.info("Getting conversation history.")

                return data_history

        except Exception as e:
            logger.debug(f"An error occurred: {str(e)}")

    def deleting_data(self):
        if not self.server:
            logger.error("Environment variable 'server_database' was not found.")
            return False

        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()

                query = """
                    WITH RecentMessages AS (
                        SELECT
                            Id,
                            ROW_NUMBER() OVER (
                                ORDER BY date_time DESC, Id DESC
                            ) AS RowNumber
                        FROM EnglishConversation
                    )
                    DELETE FROM RecentMessages
                    WHERE RowNumber > 100;
                """

                cursor.execute(query)
                deleted_rows = cursor.rowcount

                logger.info(
                    "History cleanup completed. Deleted rows: %s",
                    deleted_rows
                )

                return True

        except Exception as error:
            logger.exception(f"Error deleting old conversation messages {error}")
            return False