import sqlite3


# ---------------------------------------------------------------------
def read_db(db_file):
    '''
    read the database and fetch all data from table 
    :param db_file: name of the database file
    :return: list of tuples containing all rows from table TS
    '''
    Xy_set = []
    # Connect to the database
    try:
        with sqlite3.connect(db_file) as db:
            cursor = db.cursor()
            sql_qry_1 = "SELECT * FROM TS;"
            res = cursor.execute(sql_qry_1)
            Xy_set = res.fetchall()
    except sqlite3.Error as e:
        print(f"Database error: {e}")

    return Xy_set


if __name__ == "__main__":
    None