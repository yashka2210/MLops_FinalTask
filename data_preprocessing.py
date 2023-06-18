# Импорт библиотек
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import string
import tree_sitter
from tree_sitter import Language, Parser
import codecs


#Функция считывания файла
def file_inner(path):
    with codecs.open(path, 'r', 'utf-8') as file:
        code = file.read()
    return code


#Удаление комментариев в коде, whitespace, приведение к одной строке
def cleaner1(code):
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    code = re.sub(pat,'',code)
    code = re.sub('\r','',code)
    code = re.sub('\t','',code)
    code = code.split('\n')
    code = [line.strip() for line in code if line.strip()]
    code = ' '.join(code)
    return(code)


def subnodes_by_type(node, node_type_pattern = ''):
    if re.match(pattern=node_type_pattern, string=node.type, flags=0):
        return [node]
    nodes = []
    for child in node.children:
        nodes.extend(subnodes_by_type(child, node_type_pattern='method_declaration'))
    return nodes


def add_line_delimiter(method):
    method = method.replace(';', ';\n')
    method = method.replace('{', '\n{\n')
    method = method.replace('}', '}\n')
    return method


def obfuscate(parser, code, node_type_pattern='method_declaration'):
    code = cleaner1(code)
    tree = parser.parse(bytes(code, 'utf8'))
    nodes = subnodes_by_type(tree.root_node, node_type_pattern)
    methods = []
    for node in nodes:
        if node.start_byte >= node.end_byte:
            continue
        method = code[node.start_byte:node.end_byte]
        methods.append(add_line_delimiter(method))
    return methods


def main():
    # Загрузка датасета в Pandas
    df = pd.read_csv('data/df_cs_vuls.csv')
    # Проверим наличие дубликатов
    duplicateRows = df[df.duplicated()]
    # Удаление дубликатов (если есть)
    if len(duplicateRows) > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    # Делим данные на тренировочную, тестовую и валидационную выборки
    train, test = train_test_split(df, test_size=0.3, random_state=42)
    test, val = train_test_split(test, test_size=0.5, random_state=42)
    #Сохранение файлов
    train.to_csv('data/train.csv', index=False)
    test.to_csv('data/test.csv', index=False)
    val.to_csv('data/val.csv', index=False)

    
if __name__ == "__main__":
    main()
