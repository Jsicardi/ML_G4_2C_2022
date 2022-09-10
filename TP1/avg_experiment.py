import json
from main import __main__ as titles_main

def __main__():

    for i in range(10):
        
        with open("config.json", "r") as file:
            json_values = json.load(file)
            json_values["output_file"] = "resources/classifier_results_{0}".format(i)
            json_values["test_categories_file"] = "resources/ej2_test_categories_{0}".format(i)
        with open("config.json", "w") as file:
            json.dump(json_values,file,indent=4)
        titles_main()

if __name__ == "__main__":
    __main__()