import sqlite3

# Replace with your actual database file name
db_path = 'users.db'

# Read the SQL query from file
with open('queries.sql', 'r') as f:
    sql = f.read()

# Connect to the database and execute the query
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute(sql)
conn.commit()  # <-- This saves the changes

print("Update complete.")

conn.close()