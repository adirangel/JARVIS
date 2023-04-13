import sqlite3
from sqlite3 import Error
import json

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    if conn:
        create_table(conn)

    return conn

def create_table(conn):
    table_schema = """CREATE TABLE IF NOT EXISTS search_data (
                        query TEXT PRIMARY KEY,
                        results TEXT,
                        summaries TEXT,
                        timestamp REAL
                      );"""
    try:
        cursor = conn.cursor()
        cursor.execute(table_schema)
    except Error as e:
        print(e)


def insert_search_data(conn, query, search_results, summaries, timestamp):
    cursor = conn.cursor()
    # Convert lists to JSON strings
    search_results = json.dumps(search_results)
    summaries = json.dumps(summaries)

    cursor.execute("INSERT OR REPLACE INTO search_data (query, results, summaries, timestamp) VALUES (?, ?, ?, ?)", (query, search_results, summaries, timestamp))
    conn.commit()


def get_search_data(conn, query):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM search_data WHERE query=?", (query,))
    row = cursor.fetchone()

    if row:
        search_results = json.loads(row[1])
        summaries = row[2]
        timestamp = row[3]
        return search_results, summaries, timestamp
    else:
        return None


