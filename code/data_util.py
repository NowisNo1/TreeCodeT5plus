import json
import re


def data_transformation_for_datasets_lib(path="dataset/output.txt"):
    with open(path, 'r', encoding="UTF-8") as f:
        json_string = f.read()
        data = json.loads(json_string)
        return data


def generate_formal_data(data):
    train = []
    val = []
    for i in range(len(data)):
        sample = data[i]
        caller_name = sample["callerName"]
        caller_class = sample["callerClass"]
        caller_signature = sample["callerSignature"]
        caller_body = sample["callerBody"]

        input_string = caller_signature + caller_body

        source = input_string.replace(caller_name, "methodName")
        label = caller_name
        if i < 0.7 * len(data):
            train.append({"input": source, "target": label})
        else:
            val.append({"input": source, "target": label})

    return train, val


train_data, val_data = generate_formal_data(data_transformation_for_datasets_lib())

with open('dataset/train.json', 'w') as f:
    json.dump(train_data, f, indent=4)  # indent参数用于美化输出，使其更易读

with open('dataset/val.json', 'w') as f:
    json.dump(val_data, f, indent=4)  # indent参数用于美化输出，使其更易读


