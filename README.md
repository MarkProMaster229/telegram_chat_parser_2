# telegram_chat_parser_2
Простой пармер чатов в телеграме с контекстом разговора
# Как использовать:

1. Скачайте данный репозиторий на ваш пк
2. Установите зависимости с помощью
```
pip install -r requirements.txt
```
3. Экспортируйте чат в telegram в формат json (выбирается в меню)
4. Укажите путь к входному файлу и выходной директории
```
python parser.py --tg_history_path /path/to/history/file.json --output_path /path/to/output/directory
```
5. Ожидайте завершения
6. В указанной выходной директории будет 3 файла:
```
1. raw.csv - файл с неочищенными данными
2. train.jsonl и test.jsonl - данные готовые для дальнейшей обработки
```
P.S добавлен фильтр контента и сортировка диалога 
