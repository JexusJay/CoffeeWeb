-- schema.sql

DROP TABLE IF EXISTS user;
CREATE TABLE user (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
);

-- New table to store each analysis
DROP TABLE IF EXISTS analysis;
CREATE TABLE analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    location TEXT NOT NULL,
    coffee_type TEXT NOT NULL,
    capture_date DATE NOT NULL,
    area REAL NOT NULL,
    perimeter REAL NOT NULL,
    eccentricity REAL NOT NULL,
    solidity REAL NOT NULL,
    centroid_row REAL NOT NULL,
    centroid_col REAL NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES user (id)
);
