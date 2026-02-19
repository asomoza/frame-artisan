import logging
import os
import sqlite3
import threading


local_db = threading.local()
logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_connection(self):
        """Gets or creates a thread-local connection."""
        if not hasattr(local_db, "conn"):
            if not os.path.exists(os.path.dirname(self.db_path)):
                os.makedirs(os.path.dirname(self.db_path))
            local_db.conn = sqlite3.connect(self.db_path)
            local_db.cursor = local_db.conn.cursor()
        return local_db.conn, local_db.cursor

    def disconnect(self):
        """Disconnects the thread-local connection."""
        if hasattr(local_db, "conn"):
            local_db.conn.close()
            del local_db.conn
            del local_db.cursor

    def execute(self, query, params=None):
        """Executes a SQL query."""
        conn, cursor = self._get_connection()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            return cursor
        except sqlite3.Error as e:
            logger.error(f"An error occurred: {e}")
            conn.rollback()
            raise

    def fetch_one(self, query, params=None):
        """Fetches one row from a SQL query."""
        conn, cursor = self._get_connection()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchone()
        except sqlite3.Error as e:
            logger.error(f"An error occurred: {e}")
            raise

    def fetch_all(self, query, params=None):
        """Fetches all rows from a SQL query."""
        conn, cursor = self._get_connection()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"An error occurred: {e}")
            raise

    def create_table(self, table_name, columns):
        """Creates a table if it doesn't exist."""
        columns_str = ", ".join(columns)
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str})"
        self.execute(query)

    def insert(self, table_name, data):
        """Inserts data into a table."""
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        self.execute(query, tuple(data.values()))

    def last_insert_rowid(self):
        conn, cursor = self._get_connection()
        return cursor.lastrowid

    def update(self, table_name, data, condition):
        """Updates data in a table."""
        conn, cursor = self._get_connection()
        try:
            set_str = ", ".join([f"{key} = ?" for key in data.keys()])
            where_clause = " AND ".join([f"{key} = ?" for key in condition.keys()])
            query = f"UPDATE {table_name} SET {set_str} WHERE {where_clause}"
            params = tuple(data.values()) + tuple(condition.values())
            cursor.execute(query, params)
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"An error occurred: {e}")
            conn.rollback()
            raise

    def delete(self, table_name, condition):
        """Deletes data from a table."""
        query = f"DELETE FROM {table_name} WHERE {condition}"
        self.execute(query)

    def exists(self, table_name, column_name, value):
        """Checks if a value exists in a specific column of a table.

        Args:
            table_name: The name of the table.
            column_name: The name of the column to check.
            value: The value to search for.

        Returns:
            True if the value exists in the column, False otherwise.
        """
        conn, cursor = self._get_connection()
        try:
            query = f"SELECT 1 FROM {table_name} WHERE {column_name} = ?"
            cursor.execute(query, (value,))
            result = cursor.fetchone()
            return result is not None
        except sqlite3.Error as e:
            logger.error(f"An error occurred: {e}")
            raise

    def select(
        self, table: str, columns: list, conditions: dict = None, order_by: str = None, order_by_direction: str = "ASC"
    ):
        """
        Selects data from the database table.

        Args:
            table: Table name.
            columns: List of columns names to select.
            conditions: Dictionary of conditions for the WHERE clause.
                Example: {"id": 1, "name": "example"}
                OR
                Example: {"id": (1,2), "name": "example"} to use an IN operator.
                OR
                Example: {"name": "%example%", "LIKE": True} to use an LIKE operator.
            order_by: Order for the sql query
                Example: "name ASC" to order by name ascending.

        Returns:
            List of tuples with the selected data.
        """
        conn, cursor = self._get_connection()
        try:
            column_names = ", ".join(columns)
            query = f"SELECT {column_names} FROM {table}"

            if conditions is not None:
                where_clause = " WHERE "
                params = []
                condition_list = []
                for column, value in conditions.items():
                    if isinstance(value, tuple):
                        placeholders = ", ".join(["?"] * len(value))
                        condition_list.append(f"{column} IN ({placeholders})")
                        params.extend(value)
                    elif isinstance(value, str) and conditions.get("LIKE", False):
                        condition_list.append(f"{column} LIKE ?")
                        params.append(value)
                    else:
                        condition_list.append(f"{column} = ?")
                        params.append(value)
                where_clause += " AND ".join(condition_list)
                query += where_clause
                if conditions.get("LIKE", False):
                    if "LIKE" in conditions:
                        del conditions["LIKE"]

            if order_by is not None:
                query += f" ORDER BY LOWER({order_by}) {order_by_direction}"

            if conditions is not None:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error selecting data: {e}")
            logger.debug("Error Exception", exc_info=True)
            return []

    def select_one(self, table, columns, where):
        """Selects a single row from the database."""
        conn, cursor = self._get_connection()
        try:
            where_clause = " AND ".join([f"{k} = ?" for k in where])
            query = f"SELECT {', '.join(columns)} FROM {table} WHERE {where_clause}"
            cursor.execute(query, tuple(where.values()))
            row = cursor.fetchone()
            return dict(zip(columns, row)) if row else None
        except Exception as e:
            logger.error(f"Database error in select_one: {e}")
            return None
