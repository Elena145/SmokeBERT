def format_predictions(tokens, labels, simplified_label_map, paired_labels):
    output_str = ""
    current_label = None
    current_tokens = []
    for j in range(len(tokens)):
        token = tokens[j]
        if token.startswith("##"):
            token = token[2:]
        if token.startswith("Ġ"):
            token = token[1:]
        label = labels[j]
        if label == 0:
            if current_tokens:
                output_str += f"{simplified_label_map[current_label]}: {' '.join(current_tokens)}, "
                current_tokens = []
                current_label = None
            continue
        if current_label is not None and label == paired_labels.get(current_label):
            current_tokens.append(token)
        else:
            if current_tokens:
                output_str += f"{simplified_label_map[current_label]}: {' '.join(current_tokens)}, "
                current_tokens = []
            if label in paired_labels:
                current_label = label
                current_tokens = [token]
            else:
                current_label = label
                current_tokens = [token]
    if current_tokens:
        output_str += f"{simplified_label_map[current_label]}: {' '.join(current_tokens)}, "
    return output_str.rstrip(", ")

simplified_label_map = {
    0: 'O',
    1: 'PACKS_PER_DAY',
    2: 'PACKS_PER_DAY',
    3: 'CIGS_PER_DAY',
    4: 'CIGS_PER_DAY',
    5: 'YEARS_SMOKED',
    6: 'YEARS_SMOKED',
    7: 'PACK_YEARS',
    8: 'PACK_YEARS',
    9: 'YSQ',
    10: 'YSQ',
    11: 'QUIT_AT_YEAR',
    12: 'QUIT_AT_YEAR'
}

paired_labels = {
    1: 2,
    3: 4,
    5: 6,
    7: 8,
    9: 10,
    11: 12
}
