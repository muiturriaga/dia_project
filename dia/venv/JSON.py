import json

with open('json_data.json', 'r') as f:
    parsed_json = (json.load(f))
    # print(json.dumps(parsed_json, indent=4, sort_keys=True))


class Test(object):
    def __init__(self, data):
        self.__dict__ = data


test1 = Test(parsed_json)
print(test1)
