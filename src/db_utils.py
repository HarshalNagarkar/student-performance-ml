import sqlite3
import pandas as pd

DB_NAME = "student_performance.db"

def create_connection():
    return sqlite3.connect(DB_NAME)

def create_table(conn):
    query = """
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        attendance INTEGER,
        internal_marks INTEGER,
        assignment_score INTEGER,
        result TEXT
    );
    """
    conn.execute(query)
    conn.commit()

def insert_data(conn, df):
    df.to_sql("students", conn, if_exists="append", index=False)

if __name__ == "__main__":
    conn = create_connection()
    create_table(conn)

    df = pd.read_csv("data/student_data.csv")
    insert_data(conn, df)

    conn.close()
    print("Data successfully stored in SQL database")

