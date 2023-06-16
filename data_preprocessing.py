# Импорт библиотек
import pandas as pd
from sklearn.model_selection import train_test_split

# Загрузка датасета в Pandas
df = pd.read_csv('data/df_cs_vuls.csv')

# Проверим наличие дубликатов
duplicateRows = df[df.duplicated()]

# Удаление дубликатов (если есть)
if len(duplicateRows) > 0:
    df = df.drop_duplicates().reset_index(drop=True)

# Удаление лишних столбцов
df = df[['Snippet', 'Target']]

# Делим данные на тренировочную, тестовую и валидационную выборки
train, test = train_test_split(df, test_size=0.3, random_state=42)
test, val = train_test_split(test, test_size=0.5, random_state=42)

#Сохранение файлов
train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)
val.to_csv('data/test.csv', index=False)
