import pandas



def read_data(file_path: str) -> pandas.DataFrame:
    """
    Reads data from a CSV file and returns it as a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The data from the CSV file.
    """
    try:
        data = pandas.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None





read_data('client_log.csv')