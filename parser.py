import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from datasets import load_dataset
import orjson
import typer
import pandas as pd
import re

Message = Dict[str, Any]
Context = List[Optional[Message]]

app = typer.Typer()


BAD_WORDS = {"нежелательное_слово1", "нежелательное_слово2", "нежелательное_слово3"}

#поиск emoji
EMOJI_RE = re.compile("[\U0001F600-\U0001F64F"
                      "\U0001F300-\U0001F5FF"
                      "\U0001F680-\U0001F6FF"
                      "\U0001F1E0-\U0001F1FF]+", flags=re.UNICODE)

def contains_emoji(text: str) -> bool:
    return bool(EMOJI_RE.search(text))

def contains_bad_words(text: str) -> bool:
    lowered = text.lower()
    return any(bad in lowered for bad in BAD_WORDS)

def is_valid(text: Optional[str], min_len: int = 5, max_len: int = 500) -> bool:
    if not text:
        return False
    if not (min_len <= len(text.strip()) <= max_len):
        return False
    if contains_bad_words(text):
        return False
    if contains_emoji(text):
        return False
    return True

def export_data(path: Path):
    print(f"Loading CSV dataset from {path / 'raw.csv'}")
    data = load_dataset('csv', data_files={'train': str(path / "raw.csv")})
    data = data['train'].train_test_split(test_size=0.2)

    def is_pair_clean(sample):
        return all(is_valid(sample.get(field)) for field in ['context_1', 'response'])

    print("Filtering dataset...")
    data = data.filter(is_pair_clean)

    print("Saving train.jsonl and test.jsonl")
    with open(path / 'train.jsonl', 'wb') as dataset:
        for chunk in data['train']:
            dataset.write(orjson.dumps(chunk, option=orjson.OPT_APPEND_NEWLINE))

    with open(path / 'test.jsonl', 'wb') as dataset:
        for chunk in data['test']:
            dataset.write(orjson.dumps(chunk, option=orjson.OPT_APPEND_NEWLINE))
    print("Export finished")


@app.command()
def prepare_messages(
    tg_history_path: Path = typer.Option(..., help='Path to telegram history json file'),
    output_path: Path = typer.Option(..., help='Path to output directory'),
):
    print(f"Loading telegram history from {tg_history_path}")
    with tg_history_path.open(encoding='utf-8') as messages_file:
        data = json.load(messages_file)
    messages = data.get('messages', [])
    print(f"Loaded {len(messages)} messages")

    contexts = _create_contexts(messages)
    contexts = _transform_contexts(contexts)
    print(f"Prepared {len(contexts)} contexts")

    output_path.mkdir(parents=True, exist_ok=True)
    contexts_df = pd.DataFrame.from_records(contexts)
    contexts_df.drop_duplicates(inplace=True)
    csv_path = output_path / 'raw.csv'
    contexts_df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    export_data(output_path)

def _create_contexts(messages: List[Message]) -> List[Context]:
    replies_threads = {}
    id_to_message = {}
    for message in messages:
        id_to_message[message['id']] = message
        if 'reply_to_message_id' in message:
            replies_threads[message['reply_to_message_id']] = message['id']

    contexts = []
    cur_context = _create_default_list()
    visited_replies = set()

    for message in messages:
        if (
            message.get('type') != 'message' or
            not message.get('text') or
            not isinstance(message.get('text'), (str, list)) or
            message['id'] in visited_replies
        ):
            continue

        if 'forwarded_from' in message and cur_context:
            contexts.append(cur_context)
            cur_context = _create_default_list()
            continue

        if message['id'] in replies_threads:
            contexts.append(cur_context)
            cur_context = _create_default_list()
            _resolve_thread(contexts, replies_threads, visited_replies, id_to_message, message)
            continue

        if cur_context[-1] and message.get('from_id') == cur_context[-1].get('from_id'):
            # Объединяем текст сообщений от одного отправителя подряд
            if isinstance(cur_context[-1]['text'], list):
                # Если предыдущий текст — список, соединяем
                cur_context[-1]['text'].extend(message['text'] if isinstance(message['text'], list) else [message['text']])
            else:
                cur_context[-1]['text'] += '\n' + (message['text'] if isinstance(message['text'], str) else ''.join(message['text']))
            continue

        cur_context.pop(0)
        cur_context.append(message)
        contexts.append(cur_context.copy())

    return contexts

def _resolve_thread(
    contexts: List[Context],
    replies_threads: Dict[int, int],
    visited_replies: Set[int],
    id_to_message: Dict[int, Message],
    message: Message,
) -> None:
    cur_context = _create_default_list()
    cur_id = message['id']

    while cur_id:
        cur_context.pop(0)
        cur_context.append(id_to_message[cur_id])
        contexts.append(cur_context.copy())

        visited_replies.add(cur_id)
        cur_id = replies_threads.get(cur_id)

def _transform_contexts(contexts: List[Context]) -> List[Dict[str, Optional[str]]]:
    return [_transform_context(context) for context in contexts if any(context)]

def _transform_context(context: Context) -> Dict[str, Optional[str]]:
    return {
        'context_3': _transform_message(context[0]),
        'context_2': _transform_message(context[1]),
        'context_1': _transform_message(context[2]),
        'response': _transform_message(context[3]),
    }

def _transform_message(message: Optional[Message]) -> Optional[str]:
    if not message:
        return None

    text = message.get('text')
    if isinstance(text, list):
        texts = [text_part['text'] if isinstance(text_part, dict) else text_part for text_part in text]
        text = ''.join(texts)
    return text

def _create_default_list(message: Optional[Message] = None) -> List[Optional[Message]]:
    return [None, None, None, message]

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "prepare-messages":
        # Убираем первый аргумент, чтобы typer понял команду
        sys.argv = sys.argv[1:]
    app()
