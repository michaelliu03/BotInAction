import json

fr = open('../../data/chapter10/AnyQ_dataset.json', 'r', encoding='UTF-8')
fw = open('../../data/chapter10/AnyQ_dataset_new_2.json', 'w', encoding='UTF-8')
content = fr.readlines()
for line in content:
    load_dict = json.loads(line)
    load_dict_id =str(load_dict['id'])
    new_dict = {'question_id': load_dict_id, 'query_id': load_dict_id, 'query': load_dict['question'],
                'answer': load_dict['answer']}
    json_dict = json.dumps(new_dict, ensure_ascii=False)
    fw.write(json_dict)
    fw.write('\n')
    # print(new_dict)
fr.close()
fw.close()
