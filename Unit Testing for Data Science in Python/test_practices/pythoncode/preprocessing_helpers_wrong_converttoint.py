def convert_to_int(integer_string_with_commas):
    return 0


def row_to_list(row):
    row = row.rstrip("\n")
    separated_entries = row.split("\t")
    if len(separated_entries) == 2 and "" not in separated_entries:
        return separated_entries
    return None


def preprocess(raw_data_file_path, clean_data_file_path):
    with open(raw_data_file_path, "r") as input_file:
        rows = input_file.readlines()
    with open(clean_data_file_path, "w") as output_file:
        for row in rows:
            row_as_list = row_to_list(row)
            if row_as_list is None:
                continue
            area = convert_to_int(row_as_list[0])
            price = convert_to_int(row_as_list[1])
            if area is None or price is None:
                continue
            output_file.write("{0}\t{1}\n".format(area, price))
