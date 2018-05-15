from src.data_handler.guardian_csv import GuardianCsvData

if __name__ == '__main__':
    print("Import guardian csv data")
    guardian_csv = GuardianCsvData()

    # guardian_csv.import_article()
    # print("... articles imported")
    #
    # guardian_csv.import_authors()
    # print("... authors imported")

    guardian_csv.import_comments()
    print("... comments imported")