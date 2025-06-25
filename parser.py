import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import orjson
import pandas as pd
import typer
from datasets import load_dataset

Message = Dict[str, Any]
Context = List[Optional[Message]]

app = typer.Typer()

BAD_WORDS = {"нежелательное_слово1", "нежелательное_слово2", "нежелательное_слово3"}

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
    with open(path / 'train.jsonl', 'wb') as train_file:
        for item in data['train']:
            train_file.write(orjson.dumps(item, option=orjson.OPT_APPEND_NEWLINE))
    with open(path / 'test.jsonl', 'wb') as test_file:
        for item in data['test']:
            test_file.write(orjson.dumps(item, option=orjson.OPT_APPEND_NEWLINE))

    print("Export finished")


@app.command()
def prepare_messages(
    tg_history_path: Path = typer.Option(..., help='Path to telegram history json file'),
    output_path: Path = typer.Option(..., help='Path to output directory'),
):
    print(f"Loading telegram history from {tg_history_path}")
    with tg_history_path.open(encoding='utfФ-8') as f:
        messages = json.load(f).get("messages", [])

    print(f"Loaded {len(messages)} messages")

    contexts = _create_contexts(messages)
    transformed = _transform_contexts(contexts)

    print(f"Prepared {len(transformed)} contexts")

    output_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(transformed)
    df.drop_duplicates(inplace=True)
    csv_path = output_path / "raw.csv"
    df.to_csv(csv_path, index=False)

    print(f"Saved to {csv_path}")
    export_data(output_path)


def _create_contexts(messages: List[Message]) -> List[Context]:
    contexts: List[Context] = []
    current_context: List[Optional[Message]] = []

    last_author = None

    for msg in messages:
        if msg.get("type") != "message":
            continue
        text = msg.get("text")
        if not text or not isinstance(text, (str, list)):
            continue

        # Объединяем сообщения одного автора
        if last_author == msg.get("from_id") and current_context:
            if isinstance(current_context[-1]['text'], list):
                if isinstance(text, list):
                    current_context[-1]['text'].extend(text)
                else:
                    current_context[-1]['text'].append(text)
            else:
                if isinstance(text, list):
                    combined = ''.join(t['text'] if isinstance(t, dict) else t for t in text)
                else:
                    combined = text
                current_context[-1]['text'] += '\n' + combined
            continue

        # Новый автор — добавляем как новый шаг контекста
        current_context.append(msg)
        last_author = msg.get("from_id")

        if len(current_context) == 4:
            contexts.append(current_context.copy())
            current_context.pop(0)

    return contexts


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
    text = message.get("text")
    if isinstance(text, list):
        return ''.join(t["text"] if isinstance(t, dict) else t for t in text)
    return text


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "prepare-messages":
        sys.argv = sys.argv[1:]
    app()
