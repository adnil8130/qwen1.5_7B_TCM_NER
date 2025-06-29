import json

def exact_interval_and_label(label_list):
    intervals = []
    class_label_set = set()
    start = None
    class_label = None
    for i, label in enumerate(label_list):
        if label != 'O' and start is None:
            start = i
            class_label = label.split('-')[1]
            class_label_set.add(class_label)
            
        elif label == 'O' and start is not None:
            intervals.append((start, i-1, class_label))
            start = None
            class_label = None
    
    if start is not None:
        intervals.append((start, len(label_list)-1, class_label))

    return intervals, class_label_set

def read_text_and_generate_json_examples(file_path):
    class_sets = set()
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # 切割每句话
        content = f.read()
        sentences = content.split("\n\n")
    
        # 每句话处理出原文和对应的实体,实体信息用(start_index, end_index, class表示)
        for sentence in sentences:
            text = ""
            labels = []
            letter_labels = sentence.split("\n")
            for letter_label in letter_labels:
                letter = letter_label.split(" ")[0]
                text += letter
                label = letter_label.split(" ")[1]
                labels.append(label)
            entites_infos, class_label_set = exact_interval_and_label(labels)
            class_sets.update(class_label_set)
    
            example = dict()
            example['text'] = text
            entities = []
            for entites_info in entites_infos:
                start_index = entites_info[0]
                end_index = entites_info[1]
                entity_label = entites_info[2]
                entity = {}
                entity_text = text[start_index: end_index + 1]
                entity['start_idx'] = start_index
                entity['end_idx'] = end_index
                entity['entity_text'] = entity_text
                entity['entity_label'] = entity_label
                entities.append(entity)
                example['entities'] = entities
            examples.append(example)

    # 将examples写入json文件
    output_file_path = file_path + '.json'
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=4)
    print("output_file_path: ", output_file_path)
    print("class_sets: ", sorted(list(class_sets)))
    return output_file_path, class_sets

if __name__ == "__main__":
    data_files = './datasets'
    file_paths = [data_files + '/medical.train', data_files + '/medical.dev', data_files + '/medical.test']
    output_file_paths = []
    class_sets_merged = set()
    for file_path in file_paths:
        output_file_path, class_sets = read_text_and_generate_json_examples(file_path)
        output_file_paths.append(output_file_path)
        class_sets_merged.update(class_sets)
    print("output_file_paths: ", output_file_paths)
    print("class_sets_merged: ", sorted(list(class_sets_merged)))
    # ['中医治则', '中医治疗', '中医证候', '中医诊断', '中药', '临床表现', '其他治疗', '方剂', '西医治疗', '西医诊断']

    