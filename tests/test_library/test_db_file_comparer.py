from unittest.mock import patch

from tumorsphere.library.db_file_comparer import compare_databases


def test_compare_databases_equal(temp_db):
    schema = {
        "Cells": "CREATE TABLE Cells (id INTEGER, attr1 INTEGER, attr2 TEXT)"
    }
    data = {"Cells": [(1, 10, "A"), (2, 20, "B")]}

    db_path1 = temp_db(schema, data)
    db_path2 = temp_db(schema, data)  # Creating a second, identical temp db

    with patch("builtins.print") as mock_print:
        assert compare_databases(db_path1, db_path2) == True
        mock_print.assert_called_with("Databases are equal.")


def test_compare_databases_not_equal(temp_db):
    schema = {
        "Cells": "CREATE TABLE Cells (id INTEGER, attr1 INTEGER, attr2 TEXT)"
    }
    data_db1 = {"Cells": [(1, 10, "A"), (2, 20, "B")]}
    data_db2 = {"Cells": [(1, 10, "A"), (2, 20, "C")]}

    db_path1 = temp_db(schema, data_db1)
    db_path2 = temp_db(schema, data_db2)

    with patch("builtins.print") as mock_print:
        assert compare_databases(db_path1, db_path2) == False
        mock_print.assert_any_call("Data in table Cells is different.")
