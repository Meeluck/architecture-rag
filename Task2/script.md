# Script

## Скрипт

Файл: `Task2/fandom_html_to_markdown.py`

Скрипт умеет работать в двух режимах:

- скачать HTML по URL со страниц `gameofthrones.fandom.com`;
- взять уже скачанный локальный HTML-файл.

### Установка зависимостей

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r Task2/requirements.txt
```

### Пример запуска для одной страницы

```bash
python Task2/fandom_html_to_markdown.py \
  --url "https://gameofthrones.fandom.com/wiki/Jon_Snow" \
  --out-dir Task2/knowledge_base_raw
```

На выходе получится файл:

```text
Task2/knowledge_base_raw/jon-snow.md
```

### Пример запуска для списка страниц

Файл `Task2/got_urls.txt` уже содержит 36 страниц: персонажи, дома, организации, локации, события и объекты.

Запуск:

```bash
python Task2/fandom_html_to_markdown.py \
  --url-file Task2/got_urls.txt \
  --out-dir Task2/knowledge_base_raw \
  --sleep 3
```

По умолчанию скрипт использует облегчённый HTML Fandom через `?action=render`. Это уменьшает количество мусора на странице и снижает шанс получить `403 Forbidden`.

### Пример запуска для локального HTML

```bash
python Task2/fandom_html_to_markdown.py \
  --html-file Task2/html/Jon_Snow.html \
  --out-dir Task2/knowledge_base_raw
```

## Что делает очистка

- берёт основной блок статьи `.mw-parser-output`;
- удаляет навигацию, рекламу, инфобоксы, оглавление, сноски и служебные блоки;
- разворачивает HTML-ссылки в обычный текст;
- конвертирует заголовки, абзацы, списки и таблицы в Markdown;
- дополнительно удаляет Markdown-ссылки вида `[текст](url)`.

## Если Fandom отдаёт 403 Forbidden

Fandom может блокировать быстрые серии запросов. В скрипте для этого уже добавлены браузерные HTTP-заголовки, `requests.Session`, повторные попытки и пауза между страницами.

Рекомендуемый запуск:

```bash
python Task2/fandom_html_to_markdown.py \
  --url-file Task2/got_urls.txt \
  --out-dir Task2/knowledge_base_raw \
  --sleep 5 \
  --retries 5
```

Если нужно остановиться на первой ошибке, добавьте:

```bash
--fail-fast
```

Если `render`-режим по какой-то причине вернул неполный текст, можно попробовать полную страницу:

```bash
--fetch-mode page
```

## Извлечение сущностей для будущих замен

После подготовки Markdown-файлов можно собрать список терминов, которые нужно будет заменить на вымышленные имена, места и события.

Файл скрипта:

```text
Task2/extract_entities_from_markdown.py
```

Запуск:

```bash
python Task2/extract_entities_from_markdown.py \
  --input-dir Task2/knowledge_base_raw \
  --out-json Task2/entities.json \
  --out-md Task2/entities.md
```

На выходе будут два файла:

- `Task2/entities.json` - структурированный список для последующей генерации `terms_map.json`;
- `Task2/entities.md` - человекочитаемый отчёт, который удобно быстро проверить руками.

Скрипт выделяет четыре группы:

- `characters` - персонажи;
- `places` - места;
- `events` - события;
- `other_terms` - дома, организации, артефакты и прочие важные термины, которые тоже желательно заменить.

## Маппинг замен

Файл с подготовленными заменами:

```text
Task2/terms_map.json
```

Внутри есть тематические группы и общий `flat_map`, который удобно использовать в следующем скрипте для замены текста. При автоматической замене нужно применять пары от самой длинной исходной строки к самой короткой, чтобы, например, `House Stark` заменялся раньше, чем отдельное слово `Stark`.

## Применение маппинга к Markdown-файлам

Файл скрипта:

```text
Task2/apply_terms_map.py
```

Запуск:

```bash
python Task2/apply_terms_map.py \
  --input-dir Task2/knowledge_base_raw \
  --terms-map Task2/terms_map.json \
  --out-dir Task2/knowledge_base \
  --report Task2/replacement_report.json \
  --overwrite
```

Скрипт:

- читает исходные `.md` из `knowledge_base_raw`;
- применяет `flat_map` из `terms_map.json`;
- заменяет сначала длинные термины, потом короткие;
- сохраняет результат в `knowledge_base`;
- адаптирует имена файлов по новому заголовку документа;
- создаёт `replacement_report.json` со статистикой замен.

## Извлечение списка героев со страницы сериала

Файл скрипта:

```text
Task2/extract_cast_from_fandom.py
```

Запуск для всего раздела `Cast`:

```bash
python Task2/extract_cast_from_fandom.py \
  --url "https://gameofthrones.fandom.com/wiki/Game_of_Thrones#Retainers_at_Winterfell" \
  --out-json Task2/got_cast_characters.json \
  --out-md Task2/got_cast_characters.md
```

Если нужно вытащить только подраздел `Retainers at Winterfell`:

```bash
python Task2/extract_cast_from_fandom.py \
  --url "https://gameofthrones.fandom.com/wiki/Game_of_Thrones#Retainers_at_Winterfell" \
  --start-heading "Retainers at Winterfell" \
  --stop-at-same-level \
  --out-json Task2/got_retainers_at_winterfell.json \
  --out-md Task2/got_retainers_at_winterfell.md
```

Результат нужен для проверки текущего `terms_map.json` и поиска персонажей, которых стоит добавить в маппинг.
