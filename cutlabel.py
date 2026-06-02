import csv

input_file = "<EDGE_LABELS_CSV>"
output_file = "<EDGE_LABELS_CSV>"

count_pos = 0  # 统计 label == 1 的行数

with open(input_file, "r", newline='', encoding="utf-8") as fin, \
     open(output_file, "w", newline='', encoding="utf-8") as fout:

    # 注意：这里用逗号分隔
    reader = csv.reader(fin, delimiter=",")
    writer = csv.writer(fout, delimiter=",")

    # 读写表头
    header = next(reader, None)
    if header:
        writer.writerow(header)

    for row in reader:
        if not row:
            continue

        label_str = row[-1].strip()

        # 保证能转成整数
        try:
            label = int(label_str)
        except ValueError:
            continue

        if label == -1:
            # 全部保留
            writer.writerow(row)

        elif label == 1:
            if count_pos % 7 == 0:
                writer.writerow(row)
            count_pos += 1

        # 其他 label 自动忽略
