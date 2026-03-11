#!/usr/bin/env bash

# ------------------------------
# CSV → SQLite loader
# ------------------------------

inp_file="Dalton_octupole.csv"
db="Dalton_octupole.db"

sqlite3 "$db" <<EOF

-- Create table if it doesn't exist
CREATE TABLE IF NOT EXISTS TS (
    charge REAL,
    theta REAL,
    beta_norm REAL
);

-- Remove previous data
DELETE FROM TS;

-- Configure CSV mode
.mode csv

-- Import file
.import $inp_file TS

EOF
