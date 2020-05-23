import os
import xml.etree.ElementTree as ET
import re
def read_files(path, category, n,re_rules):
    files = os.listdir(path)

    paths = []
    for file in files:
        if not os.path.isdir(os.path.join(os.path.abspath(path), file)):
            paths.append(os.path.join(path, file))
    with open(os.path.join(os.getcwd() ,category)+".txt",'w') as file:
        for i in range(0, len(paths), n):
            for j in range(i, min(n+i, len(paths))):
                tree = ET.parse(os.path.join(path, files[j]))
                root = tree.getroot()
                for child in root.iter(category):
                    flag = True
                    for re_rule in re_rules:
                        if re.search(re_rule, child.attrib["text"]) is not None:
                            flag = False
                            break
                    if flag:
                        file.write(child.attrib["text"])
                        file.write("\n")


if __name__ == "__main__":
    re_DATE1 = r'^\d{4}-(0[1-9]|1[0-2])-([0-2][0-9]|3[0-1])$'
    re_DATE2 = r'^\d{4}$'
    re_DATE3 = r'^([0-9]|1[0-2]|0[0-9])-([0-9]|[1-2][0-9]|3[0-1]|0[0-9])-(\d{4}|\d{2})$'
    re_DATE4 = r'^([0-9]|1[0-2]|0[0-9])/([0-9]|[1-2][0-9]|3[0-1]|0[0-9])/(\d{4}|\d{2})$'
    re_DATE5 = r'^([0-9]|1[0-2])/\'\d{2}$'
    re_DATE6 = r'^\'\d{2}$'
    re_DATE7 = r'^(\d{4}|\d{2})\'s$'
    re_DATE8 = r'^(\d{4}|\d{2})s$'
    re_DATE9 = r'^([0-9]|1[0-2]|0[0-9])\.([0-9]|[1-2][0-9]|3[0-1]|0[0-9])\.(\d{2})$'
    re_DATE10 = r'^([0-9]|[1-2][0-9]|3[0-1]|0[0-9])([ A-Za-z]+)(\d{4}|\d{2})$'
    re_DATE11 = r'^([ A-Za-z]+)(\d{4}|\d{2})$'
    re_DATE12 = r'^([0-9]|1[0-2]|0[0-9])/(\d{2,4})$'
    re_DATE13 = r'^((Monday){1}|(Sunday){1}|(Tuesday){1}|(Wednesday){1}|(Thursday){1}|(Friday){1}|(Saturday){1})s?$'
    re_DATE14 = r'^(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)$'
    re_month = "(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    re_DATE15 = r'^(\d{2}-'+ re_month + r'-\d{4})$'
    re_DATE16 = r'^\d{1}/\d{1}$'
    re_DATE17 = r'^((3rd)|(\dth))$'
    re_is_2_digit = r'^\d{1,2}$'
    re_is_comma = r'^[,]$'

    rules =[]
    for j in range(1, 12):
        a = "re_DATE" + str(j)
        rules.append(eval(a))

    read_files("/home/lightspace/Documents/course/npl/project/training-PHI-Gold-Set1", "DATE", 1000,rules)
