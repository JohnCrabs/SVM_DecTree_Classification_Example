import pandas as pd


def read_xls_file(filePath, sheetName):
    try:
        # Confirm file exists.
        data = pd.read_excel(filePath, sheetName)
        print(' .. successful parsing of file:', filePath)
        print("Column headings:")
        print(data.columns)
        print()
        return data
    except FileNotFoundError:
        print(FileNotFoundError)


if __name__ == '__main__':
    FILE_PATH = "data/iris.xls"
    SHEET_NAME = "Iris"
    read_xls_file(FILE_PATH, SHEET_NAME)
