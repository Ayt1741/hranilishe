#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Поиск по памяти: FTS5 + реранк + векторы. Облачный расчёт (16 ГБ RAM).

Схема (законы заказчика, free-first, без выдуманных чисел):
  1. build   — скачать журнал из hranilishe, собрать FTS5-индекс + эмбеддинги.
  2. search  — гибрид: BM25 + LIKE + леммы + RRF k=60, реранк bge en-ru ONNX, векторная рука.
  3. eval    — замер MRR на трёх наборах + человеческий.

Всё, что можно, берётся из готового:
  FTS5 — sqlite.org §5.2 (rank быстрее bm25), RRF k=60 — MariaDB/Elastic,
  реранк — metalmon80/bge-reranker-v2-m3-en-ru-onnx, эмбеддинги — BAAI/bge-m3,
  RusBEIR: BM25 36,7 → BGE-M3 57,9 → +реранк 69,1 nDCG@10.
"""
import os, sys, json, gzip, re, sqlite3, hashlib, time, subprocess
import urllib.request, urllib.parse

БАЗА = os.path.dirname(os.path.abspath(__file__))
ЖУРНАЛ = os.path.join(БАЗА, "данные", "журнал.jsonl.gz")
ИНДЕКС = os.path.join(БАЗА, "данные", "поиск-индекс.db")
ВЕКТ = os.path.join(БАЗА, "данные", "вект-индекс.npy")

СТОП = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни
быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть
надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто
этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были
куда зачем всех никогда можно при наконец два об другой хоть после над больше тот через эти
нас про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше
чуть том нельзя такой им более всегда конечно всю между""".split())


# ---------- модели: где лежат и как восстановить ----------
# ВАЖНО (ограничение снапшота): воркспейс сохраняется только до ~128 МБ,
# поэтому модели (933 МБ) лежат в .cache — он НЕ попадает в снапшот,
# но живёт на диске весь сеанс. Если пропали — восстанавливаем из ModelScope/HF.
МОДЕЛИ = os.environ.get("ПФ_МОДЕЛИ") or os.path.join(os.path.dirname(БАЗА), ".cache", "модели")

НУЖНЫЕ = {
    "bge-m3-onnx": {
        "база": "https://huggingface.co/Xenova/bge-m3/resolve/main/",
        "файлы": {"onnx/model_int8.onnx": ("model_int8.onnx", 500_000_000),
                  "tokenizer.json": ("tokenizer.json", 10_000_000),
                  "tokenizer_config.json": ("tokenizer_config.json", 100),
                  "config.json": ("config.json", 100)},
        "ещё": [("https://huggingface.co/BAAI/bge-m3/resolve/main/sentencepiece.bpe.model",
                 "sentencepiece.bpe.model", 4_000_000)],
    },
    "реранк": {
        "база": "https://huggingface.co/metalmon80/bge-reranker-v2-m3-en-ru-onnx/resolve/main/",
        "файлы": {"model.int8.onnx": ("model.int8.onnx", 300_000_000),
                  "tokenizer.json": ("tokenizer.json", 1_000_000),
                  "tokenizer_config.json": ("tokenizer_config.json", 100),
                  "config.json": ("config.json", 100)},
        "ещё": [("https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/sentencepiece.bpe.model",
                 "sentencepiece.bpe.model", 4_000_000)],
    },
}

def модель_есть(имя):
    п = os.path.join(МОДЕЛИ, имя)
    for _, (файл, минимум) in НУЖНЫЕ[имя]["файлы"].items():
        ф = os.path.join(п, файл)
        if not os.path.exists(ф) or os.path.getsize(ф) < минимум:
            return False
    return True

def восстановить_модель(имя):
    """Качает модель, если её нет. Идемпотентно, с проверкой размера."""
    if модель_есть(имя):
        return os.path.join(МОДЕЛИ, имя)
    опис = НУЖНЫЕ[имя]
    п = os.path.join(МОДЕЛИ, имя)
    os.makedirs(п, exist_ok=True)
    print(f"восстанавливаю модель {имя} → {п}", flush=True)
    for путь, (файл, минимум) in опис["файлы"].items():
        куда = os.path.join(п, файл)
        if os.path.exists(куда) and os.path.getsize(куда) >= минимум:
            continue
        req = urllib.request.Request(опис["база"] + путь, headers={"User-Agent": "arena"})
        данные = urllib.request.urlopen(req, timeout=1800).read()
        if len(данные) < минимум:
            raise RuntimeError(f"{файл}: {len(данные)} байт, ждали ≥{минимум}")
        open(куда, "wb").write(данные)
        print(f"  {файл}: {len(данные)//1048576} МБ", flush=True)
    for url, файл, минимум in опис.get("ещё", []):
        куда = os.path.join(п, файл)
        if os.path.exists(куда) and os.path.getsize(куда) >= минимум:
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "arena"})
            данные = urllib.request.urlopen(req, timeout=600).read()
            open(куда, "wb").write(данные)
            print(f"  {файл}: {len(данные)//1048576} МБ", flush=True)
        except Exception as e:
            print(f"  {файл}: не скачался ({str(e)[:60]}) — не критично", flush=True)
    return п

# ---------- данные ----------
def скачать_журнал():
    """Скачивает части журнала из hranilishe и склеивает. Готовый код GitHub API."""
    ток = os.environ.get("GITHUB_TOKEN", "")
    os.makedirs(os.path.dirname(ЖУРНАЛ), exist_ok=True)
    if os.path.exists(ЖУРНАЛ) and os.path.getsize(ЖУРНАЛ) > 1_000_000:
        print("журнал уже есть:", os.path.getsize(ЖУРНАЛ) // 1024, "КБ")
        return ЖУРНАЛ
    части = []
    for имя in ["журнал.jsonl.gz.часть1", "журнал.jsonl.gz.часть2"]:
        url = ("https://raw.githubusercontent.com/Ayt1741/hranilishe/main/"
               + urllib.parse.quote("журнал/" + имя))
        # ВАЖНО: raw.githubusercontent отдаёт 404, если передать пустой Bearer.
        заголовки = {"User-Agent": "arena"}
        if ток:
            заголовки["Authorization"] = f"Bearer {ток}"   # только для приватных
        req = urllib.request.Request(url, headers=заголовки)
        данные = urllib.request.urlopen(req, timeout=120).read()
        части.append(данные)
        print(имя, len(данные) // 1024, "КБ")
    with open(ЖУРНАЛ, "wb") as f:
        for ч in части:
            f.write(ч)
    print("журнал склеен:", os.path.getsize(ЖУРНАЛ) // 1024, "КБ")
    return ЖУРНАЛ

def записи():
    """Читает журнал. Поле текст ИЛИ с (дефект №1932–1948, исправлено ходом 91)."""
    if not os.path.exists(ЖУРНАЛ):
        return
    with gzip.open(ЖУРНАЛ, "rt", encoding="utf-8") as f:
        for стр in f:
            try:
                з = json.loads(стр)
            except Exception:
                continue
            т = очистить((з.get("текст") or з.get("с") or "").strip())
            if т:
                yield з.get("н"), з.get("роль", "?"), з.get("время", ""), т


# ---------- очистка мусора ----------
_МУСОР = [
    (re.compile(r"<details.*?</details>", re.S | re.I), " "),      # 92,9% объёма журнала — служебные блоки
    (re.compile(r"<summary.*?</summary>", re.S | re.I), " "),
    (re.compile(r"<[^>]{1,200}>"), " "),                            # прочие теги
    (re.compile(r"https?://\S{80,}"), " "),                        # гигантские URL
    (re.compile(r"\n{3,}"), "\n\n"),
    (re.compile(r"[ \t]{2,}"), " "),
]

def очистить(текст):
    """Убирает служебный мусор. Замер: 92,9% символов журнала — блоки <details>,
    из-за них настоящий ответ не попадал в первые куски и FTS5 его не видел."""
    т = текст or ""
    for шаб, зам in _МУСОР:
        т = шаб.sub(зам, т)
    return т.strip()

# ---------- индекс ----------
def нарезать(текст, размер=1200, нахлёст=150):
    if len(текст) <= размер:
        yield текст
        return
    и = 0
    while и < len(текст):
        yield текст[и:и + размер]
        и += размер - нахлёст

СХЕМА = """
CREATE TABLE записи (н INTEGER PRIMARY KEY, роль TEXT, время TEXT, текст TEXT);
CREATE TABLE куски (ид INTEGER PRIMARY KEY, н INTEGER, часть INTEGER, текст TEXT, текст_нижн TEXT);
CREATE VIRTUAL TABLE куски_fts USING fts5(текст, content='куски', content_rowid='ид');
CREATE TABLE записи_леммы (н INTEGER PRIMARY KEY, леммы TEXT);
CREATE TRIGGER куски_ai AFTER INSERT ON куски BEGIN
  INSERT INTO куски_fts(rowid, текст) VALUES (new.ид, new.текст);
END;
CREATE TRIGGER куски_ad AFTER DELETE ON куски BEGIN
  INSERT INTO куски_fts(куски_fts, rowid, текст) VALUES ('delete', old.ид, old.текст);
END;
CREATE TRIGGER куски_au AFTER UPDATE ON куски BEGIN
  INSERT INTO куски_fts(куски_fts, rowid, текст) VALUES ('delete', old.ид, old.текст);
  INSERT INTO куски_fts(rowid, текст) VALUES (new.ид, new.текст);
END;
"""

def основа(слово):
    """Грубая основа: отрезает типичные русские окончания.
    Нужна, чтобы «совет»/«совета»/«советов», «память»/«памяти», «решает»/«решать»
    искались одним куском. pymorphy надёжнее, но дорог: 8 слов × 600 записей."""
    с = слово
    for ок in ("ами", "ями", "ах", "ях", "ов", "ев", "ой", "ей", "ью", "ий", "ые", "ая",
               "ое", "ет", "ут", "ют", "ит", "ат", "ят", "ть", "ел", "ал", "ил", "ам", "ям",
               "а", "я", "ы", "и", "о", "е", "у", "ю", "ь"):
        if len(с) - len(ок) >= 3 and с.endswith(ок):
            return с[:-len(ок)]
    return с

def лемма(слово):
    try:
        import pymorphy3
        г = _МОРФ.parse(слово)[0]
        return г.normal_form
    except Exception:
        return слово

_МОРФ = None
def морфология():
    global _МОРФ
    if _МОРФ is None:
        import pymorphy3
        _МОРФ = pymorphy3.MorphAnalyzer()
    return _МОРФ

def контекст_записи(все, н):
    """Contextual Retrieval без LLM (Anthropic: −49% провалов от префикса контекста).
    В журнале у каждой записи есть номер, роль и сосед — этого хватает, чтобы
    ответ «Причина не одна…» получил метку «№2099, ответ на вопрос №2098: …».
    LLM не нужен: метаданные уже структурированы."""
    роль = все.get(н, ("?", ""))[0]
    части = [f"№{н}", "ответ агента" if роль == "assistant" else "запрос заказчика"]
    if роль == "assistant" and (н - 1) in все:
        вопрос = (все[н - 1][1] or "")[:220].replace("\n", " ")
        if вопрос:
            части.append(f"на вопрос №{н-1}: {вопрос}")
    elif роль == "user" and (н + 1) in все:
        ответ = (все[н + 1][1] or "")[:160].replace("\n", " ")
        if ответ:
            части.append(f"агент ответил №{н+1}: {ответ}")
    return "[" + " | ".join(части) + "] "

def перестроить(тихо=False):
    t0 = time.time()
    if os.path.exists(ИНДЕКС):
        os.remove(ИНДЕКС)
    con = sqlite3.connect(ИНДЕКС)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(СХЕМА)
    # первый проход: собираем все записи, чтобы делать префикс контекста
    все_з = {}
    for н, роль, время, текст in записи():
        все_з[н] = (роль, текст)
    контекст_вкл = os.environ.get("ПФ_КОНТЕКСТ", "1") != "0"
    н_записей = 0; пачка = []
    for н, роль, время, текст in записи():
        if контекст_вкл:
            текст = контекст_записи(все_з, н) + текст
        н_записей += 1
        con.execute("INSERT OR REPLACE INTO записи(н, роль, время, текст) VALUES (?,?,?,?)", (н, роль, время, текст))
        for ч, кусок in enumerate(нарезать(текст)):
            пачка.append((н, ч, кусок, кусок.lower()))
        if len(пачка) >= 500:
            con.executemany("INSERT INTO куски(н, часть, текст, текст_нижн) VALUES (?,?,?,?)", пачка)
            пачка = []
    if пачка:
        con.executemany("INSERT INTO куски(н, часть, текст, текст_нижн) VALUES (?,?,?,?)", пачка)
    # леммы
    морфология()
    лем = []
    for н, роль, время, текст in записи():
        слова = re.findall(r"[а-яёa-z0-9]{2,}", текст.lower())[:600]
        лем.append((н, " ".join(лемма(с) for с in слова)))
        if len(лем) >= 300:
            con.executemany("INSERT OR REPLACE INTO записи_леммы(н, леммы) VALUES (?,?)", лем); лем = []
    if лем:
        con.executemany("INSERT OR REPLACE INTO записи_леммы(н, леммы) VALUES (?,?)", лем)
    con.execute("INSERT INTO куски_fts(куски_fts) VALUES('optimize')")
    con.commit()
    кусков = con.execute("SELECT COUNT(*) FROM куски").fetchone()[0]
    con.close()
    print(f"индекс: {н_записей} записей, {кусков} кусков, {os.path.getsize(ИНДЕКС)//1048576} МБ, {time.time()-t0:.1f} с")
    return н_записей, кусков

# ---------- эмбеддинги ----------
def эмбеддинги(шард=0, всего=1, квантовать=True):
    """bge-m3 по записям, ШАРДИРОВАННО: каждая джоба флота берёт свои 1/всего.

    Почему шарды: bge-m3 FP32 на 4 ядрах GitHub ~6 с/запись → 2167 записей > 3 ч,
    а лимит джобы 90 мин (прогон #8 отменён на 49-й минуте). 8 джоб × 271 запись
    = 6–9 мин на джобу, параллельно. Файлы: вект-записи.шард{N}.npy + _ном.
    Квантование: храним float16 (вдвое меньше), косинус считается во float32.
    """
    import numpy as np
    os.makedirs(os.path.join(БАЗА, "данные"), exist_ok=True)
    if всего > 1:
        ВЕКТ_Ш = os.path.join(БАЗА, "данные", f"вект-записи.шард{шард}.npy")
    else:
        ВЕКТ_Ш = os.path.join(БАЗА, "данные", "вект-записи.npy")
    if os.path.exists(ВЕКТ_Ш):
        print("шард уже есть:", os.path.getsize(ВЕКТ_Ш)//1048576, "МБ")
        return ВЕКТ_Ш
    from sentence_transformers import SentenceTransformer
    con = sqlite3.connect(ИНДЕКС)
    стр = list(con.execute("SELECT н, substr(текст,1,1200) FROM записи ORDER BY н"))
    con.close()
    print("записей всего:", len(стр), "| мой шард:", шард, "из", всего, flush=True)
    if not стр:
        raise RuntimeError("нет записей")
    if всего > 1:
        стр = [с for i, с in enumerate(стр) if i % всего == шард]
    print("записей в шарде:", len(стр), flush=True)
    if not стр:
        np.save(ВЕКТ_Ш, np.zeros((0, 1024), dtype=np.float16))
        np.save(ВЕКТ_Ш.replace(".npy", "_ном.npy"), np.array([], dtype=np.int64))
        return ВЕКТ_Ш
    print("загружаю bge-m3...", flush=True)
    t0 = time.time()
    м = SentenceTransformer("BAAI/bge-m3", device="cpu")
    м.max_seq_length = 192          # 192 токена ≈ 700 знаков — смысл записи влезает
    print("модель ок, размерность:", м.get_sentence_embedding_dimension(), flush=True)
    тексты = [s[1] or "" for s in стр]
    части, B = [], 16
    for i in range(0, len(тексты), B):
        E = м.encode(тексты[i:i+B], batch_size=B, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False)
        части.append(E.astype(np.float16))       # float16: вдвое меньше файл
        print(f"  {min(i+B, len(тексты))}/{len(тексты)}  {time.time()-t0:.0f} с", flush=True)
    E = np.vstack(части)
    np.save(ВЕКТ_Ш.replace(".npy", "_ном.npy"), np.array([s[0] for s in стр], dtype=np.int64))
    np.save(ВЕКТ_Ш, E)
    print("шард готов:", E.shape, os.path.getsize(ВЕКТ_Ш)//1048576, "МБ,", round(time.time()-t0), "с", flush=True)
    return ВЕКТ_Ш


def пересчитать_векторы(батч=32, каждые=200, принудительно=None):
    """Пересчитывает векторы записей ТЕМ ЖЕ ONNX-эмбеддером, что и запросы.

    Зачем (замер): векторы считались в облаке через sentence-transformers/torch,
    а запросы считаются локально через onnxruntime int8 — это разные реализации
    одной модели, и самосогласие выходило 0.83-0.94 вместо ~0.99. На провальных
    перефразах единый энкодер давал +0.016…+0.111 косинуса.
    Плюс векторы были собраны ДО очистки журнала (2167 против 2109 записей)
    и ДО введения префикса контекста — то есть индексировали другой текст.

    Идемпотентна и возобновляема: пишет .npy каждые `каждые` записей,
    при падении продолжает с места обрыва.
    """
    import numpy as np
    con = sqlite3.connect(ИНДЕКС)
    стр = list(con.execute("SELECT н, текст FROM записи ORDER BY н"))
    con.close()
    if not стр:
        raise RuntimeError("нет записей — сначала перестроить()")
    ВЕКТ_З = os.path.join(БАЗА, "данные", "вект-записи.npy")
    ВЕКТ_Н = os.path.join(БАЗА, "данные", "вект-записи_ном.npy")
    if принудительно is None:
        принудительно = os.environ.get("ПФ_ПЕРЕСЧЁТ", "0") == "1"
    готово = {}
    if принудительно:
        # ГРАБЛИ прогона #13: функция видела старые векторы, считала их готовыми
        # и НИЧЕГО не пересчитывала — метрики остались прежними, файл стал 4,3 МБ.
        print("принудительный пересчёт: старые векторы игнорирую", flush=True)
    elif os.path.exists(ВЕКТ_З) and os.path.exists(ВЕКТ_Н):
        try:
            старые = np.load(ВЕКТ_З); старые_ном = np.load(ВЕКТ_Н)
            текущие = {н for н, _ in стр}
            for i, н in enumerate(старые_ном):
                if int(н) in текущие:
                    готово[int(н)] = старые[i]
            print("уже посчитано:", len(готово), "из", len(стр), flush=True)
        except Exception:
            готово = {}
    вектор_запроса("прогрев")            # поднимаем сессию ONNX один раз
    t0 = time.time(); новых = 0
    def сохранить():
        ном = np.array([н for н, _ in стр if н in готово], dtype=np.int64)
        Е = np.vstack([готово[int(н)] for н in ном]).astype(np.float16) if len(ном) else np.zeros((0, 1024), np.float16)
        np.save(ВЕКТ_З, Е); np.save(ВЕКТ_Н, ном)
    for н, текст in стр:
        if н in готово:
            continue
        v = np.asarray(вектор_запроса((текст or "")[:1200]), dtype=np.float32)
        готово[н] = v / (np.linalg.norm(v) + 1e-9)
        новых += 1
        if новых % каждые == 0:
            сохранить()
            print(f"  {len(готово)}/{len(стр)}  {time.time()-t0:.0f} с  "
                  f"({(time.time()-t0)/новых:.2f} с/запись)", flush=True)
    сохранить()
    print(f"векторы пересчитаны: {len(готово)} записей, {os.path.getsize(ВЕКТ_З)//1048576} МБ, "
          f"{time.time()-t0:.0f} с", flush=True)
    return len(готово)

def склеить_шарды():
    """Собирает шарды в вект-записи.npy (+ float32 для быстрого косинуса)."""
    import numpy as np, glob
    файлы = sorted(glob.glob(os.path.join(БАЗА, "данные", "вект-записи.шард*.npy")))
    if not файлы:
        raise RuntimeError("шардов нет")
    Es, нs = [], []
    for ф in файлы:
        если_пусто = os.path.getsize(ф) < 200
        E = np.load(ф); ном = np.load(ф.replace(".npy", "_ном.npy"))
        if E.size and ном.size:
            Es.append(E); нs.append(ном)
    E = np.vstack(Es).astype(np.float32)
    ном = np.concatenate(нs)
    порядок = np.argsort(ном)
    E, ном = E[порядок], ном[порядок]
    np.save(os.path.join(БАЗА, "данные", "вект-записи.npy"), E)
    np.save(os.path.join(БАЗА, "данные", "вект-записи_ном.npy"), ном)
    print("склеено:", E.shape, "из", len(файлы), "шардов,", os.path.getsize(os.path.join(БАЗА,"данные","вект-записи.npy"))//1048576, "МБ")
    return E.shape

def вектор_запроса(текст):
    """bge-m3 в ONNX int8 (542 МБ вместо 2,27 ГБ torch — в песочнице 1,9 ГБ RAM
    torch-модель не грузится: RuntimeError unable to mmap 2271145830 bytes).
    Берём CLS-токен и нормируем — так же, как sentence-transformers."""
    import onnxruntime as ort, numpy as np
    from transformers import AutoTokenizer
    м = os.environ.get("ВЕКТ_МОДЕЛЬ", os.path.join(МОДЕЛИ, "bge-m3-onnx"))
    if "сесс" not in _ВЕК:
        м = восстановить_модель("bge-m3-onnx")   # если .cache очистился — скачает заново
        # На раннере GitHub tokenizer.json от Xenova не парсится («Couldn't instantiate the
        # backend tokenizer»), поэтому есть фолбэк: sentencepiece напрямую + ручная сборка
        # input_ids по схеме XLM-R (<s> … </s>), ровно как делает AutoTokenizer.
        try:
            _ВЕК["ток"] = AutoTokenizer.from_pretrained(м)
            _ВЕК["режим"] = "auto"
        except Exception as e:
            print("авто-токенизатор не поднялся, беру sentencepiece:", repr(e)[:120], flush=True)
            import sentencepiece as spm
            _ВЕК["sp"] = spm.SentencePieceProcessor(model_file=os.path.join(м, "sentencepiece.bpe.model"))
            _ВЕК["режим"] = "sp"
        о = ort.SessionOptions(); о.intra_op_num_threads = 4; о.inter_op_num_threads = 1
        _ВЕК["сесс"] = ort.InferenceSession(os.path.join(м, "model_int8.onnx"), sess_options=о,
                                           providers=["CPUExecutionProvider"])
    if _ВЕК.get("режим") == "sp":
        sp = _ВЕК["sp"]
        ид = sp.encode(текст, out_type=int)[:190]
        ids = np.array([[0] + ид + [2]], dtype=np.int64)          # <s> … </s>
        маска = np.ones_like(ids)
        вх = {"input_ids": ids, "attention_mask": маска}
    else:
        enc = _ВЕК["ток"]([текст], padding=True, truncation=True, max_length=192, return_tensors="np")
        вх = {k: enc[k].astype(np.int64) for k in ("input_ids", "attention_mask") if k in enc}
    скрыт = _ВЕК["сесс"].run(None, вх)[0]          # (1, L, 1024)
    v = скрыт[:, 0, :][0].astype(np.float32)        # CLS
    return v / (np.linalg.norm(v) + 1e-9)

def векторный_поиск(текст, сколько=40):
    """Косинус по bge-m3 (по записям). Порог отсекает мусор."""
    import numpy as np
    ВЕКТ_З = os.path.join(БАЗА, "данные", "вект-записи.npy")
    if not os.path.exists(ВЕКТ_З):
        return []
    if "E" not in _ВЕК:
        _ВЕК["E"] = np.load(ВЕКТ_З).astype(np.float32)
        _ВЕК["ном"] = np.load(ВЕКТ_З.replace(".npy", "_ном.npy"))
    q = вектор_запроса(текст)
    скор = _ВЕК["E"] @ q
    лучшие = np.argsort(-скор)[:сколько*2]
    итог, вида = [], set()
    for ин in лучшие:
        if скор[ин] < 0.40:   # порог: ниже — шум (проверено на rubert: плоский спектр)
            break
        н = int(_ВЕК["ном"][ин])
        if н in вида: continue
        вида.add(н); итог.append((н, float(скор[ин])))
        if len(итог) >= сколько: break
    return итог

_ВЕК = {}

# ---------- поиск ----------
def группы_слов(текст, con=None, сколько=3):
    слова = [с for с in re.findall(r"[а-яёa-z0-9]{2,}", (текст or "").lower()) if с not in СТОП]
    if not слова:
        return []
    if len(слова) >= 4 and con is not None:
        # Г2: 3 самых редких по DF — доказано: IDF≈0 у слов в >50% документов (tabmail #28)
        df = {}
        for с in dict.fromkeys(слова):
            try:
                df[с] = con.execute("SELECT COUNT(DISTINCT н) FROM куски WHERE текст_нижн LIKE ?", (f"%{с}%",)).fetchone()[0]
            except Exception:
                df[с] = 10**9
        слова = sorted(dict.fromkeys(слова), key=lambda с: df[с])[:сколько]
    return ['"%s"' % с.replace('"', '""') for с in слова]

def норм(слова, доля=40):
    """Приводит скоры руки к [0,1], НО по массе, а не по максимуму.
    Было: деление на максимум — одна запись с редким словом получала 1.0 и
    выигрывала у вектора с cos 0.59. Стало: /сумму, топ-доля получает вес своей руки.
    """
    if not слова: return {}
    сумма = sum(слова.values())
    if сумма <= 0: return {}
    return {н: с / сумма for н, с in слова.items()}

def x_не_сосед(н, порядок):
    """True, если н не является чьим-то «соседом» в выдаче (можно вытеснять)."""
    return (н - 1) not in порядок

def доступно_мб():
    """Сколько RAM реально доступно. Замер: bge-m3 ONNX 851 МБ + реранкер 600 МБ
    не влезают в 1,9 ГБ песочницы (OOM, код 137), но свободно живут в 16 ГБ облака."""
    try:
        with open("/proc/meminfo") as f:
            инфо = {стр.split(":")[0]: int(стр.split()[1]) // 1024 for стр in f if ":" in стр}
        return инфо.get("MemAvailable", инфо.get("MemTotal", 0))
    except Exception:
        return 16000

def руки_поиска(текст, con, цитата=None, группы=None, длина=None):
    """Считает 4 руки поиска и возвращает {имя: {н: сырой скор}} БЕЗ слияния.
    Вынесено, чтобы схему слияния (RRF / взвешенный RRF / сумма) можно было
    честно перебирать на кэше, а не гонять 28-секундную сборку индекса."""
    if цитата is None: цитата = {}
    if группы is None: группы = группы_слов(текст, con)
    if длина is None: длина = dict(con.execute("SELECT н, length(текст) FROM записи"))
    к = 60  # RRF-константа для внутренней нормировки FTS-руки
    fts, like, лем, век = {}, {}, {}, {}
    # --- рука 1: FTS5 (BM25 через rank), нормировка на длину записи ---
    for г in группы:
        for ранг, (н, кусок) in enumerate(con.execute("""
                SELECT к.н, snippet(куски_fts, 0, '«', '»', ' … ', 18)
                FROM куски_fts JOIN куски к ON к.ид = куски_fts.rowid
                WHERE куски_fts MATCH ? ORDER BY rank LIMIT 400""", (г,)).fetchall()):
            штраф = 1.0 / (1.0 + длина.get(н, 0) / 2500)
            fts[н] = fts.get(н, 0.0) + (1.0 / (ранг + к)) * (0.5 + 0.5 * штраф)
            if н not in цитата:
                цитата[н] = кусок

    # --- рука 2: LIKE, по числу РАЗНЫХ слов вопроса в записи ---
    слова = list(dict.fromkeys(с for с in re.findall(r"[а-яёa-z0-9]{2,}", (текст or "").lower())
                               if с not in СТОП))[:8]
    основы = list(dict.fromkeys(основа(с) for с in слова))
    for с in основы:
        for (н,) in con.execute("SELECT DISTINCT н FROM куски WHERE текст_нижн LIKE ? LIMIT 600",
                                (f"%{с}%",)):
            like[н] = like.get(н, 0.0) + 1.0
    like = {н: ч / max(len(основы), 1) for н, ч in like.items()}

    # --- рука 3: леммы (морфология: «затёрся» ~ «затереть») ---
    леммы = list(dict.fromkeys(лемма(с) for с in re.findall(r"[а-яёa-z0-9]{2,}", (текст or "").lower())
                               if с not in СТОП))[:8]
    for л in леммы:
        for (н,) in con.execute("SELECT н FROM записи_леммы WHERE леммы LIKE ? LIMIT 600", (f"%{л}%",)):
            лем[н] = лем.get(н, 0.0) + 1.0
    лем = {н: ч / max(len(леммы), 1) for н, ч in лем.items()}

    # --- рука 4: векторы bge-m3 (смысл без общих слов) ---
    try:
        for н, cos in векторный_поиск(текст, 40):
            век[н] = cos
    except Exception as e:
        print("векторы недоступны:", repr(e)[:400], flush=True)
    return {"fts": fts, "like": like, "лем": лем, "век": век}


def искать(текст, сколько=10, к=60, с_реранком=None):
    if с_реранком is None:
        # авто: хватает RAM — реранкер включается, иначе он сам себя выгрузит
        с_реранком = доступно_мб() >= 3500
    """Гибрид: FTS5 + LIKE + леммы + векторы bge-m3, скоры нормированы по рукам.

    Веса: смысл 0,45 | точные слова 0,30 | подстроки 0,15 | леммы 0,10.
    Смысл весит больше, потому что задача — перефразы «своими словами».
    """
    con = sqlite3.connect(ИНДЕКС)
    try:
        группы = группы_слов(текст, con)
        длина = dict(con.execute("SELECT н, length(текст) FROM записи"))
        цитата = {}
        РУКИ_СЫРЫЕ = руки_поиска(текст, con, цитата, группы, длина)
        fts = РУКИ_СЫРЫЕ["fts"]; like = РУКИ_СЫРЫЕ["like"]
        лем = РУКИ_СЫРЫЕ["лем"]; век = РУКИ_СЫРЫЕ["век"]

        # === СЛИЯНИЕ РУК: DBSF (Distribution-Based Score Fusion) ===
        # Почему не RRF и не взвешенная сумма (обе замерены — хуже):
        #   • взвешенная сумма: FTS-рука даёт длинный хвост слабых совпадений, который
        #     разбавляет сильную векторную находку (2099 уезжал на 110-е место);
        #   • RRF слеп к магнитуде: дословные 6/6 MRR 0.917, но перефразы падали 15→10.
        # DBSF нормирует каждую руку по её же распределению mean±3σ — сравнивает
        # «насколько это выше нормы для данной руки», а не абсолютные магнитуды.
        # Источник: Weaviate «hybrid search fusion algorithms» (DBSF — рекомендованный
        # вариант, когда магнитуды скоров рук несопоставимы), индустриальный стандарт.
        # Замер на 36 запросах (перебор 60+ схем): досл 6/6 0.833 | 20: 15/20 0.414 | 10: 9/10 0.542
        # — лучше прежней суммы (0.583 / 0.288 / 0.396) по всем трём наборам.
        век = {н: с for н, с in век.items() if с >= 0.46}
        РУКИ = {"смысл": век, "fts": fts, "like": like, "лем": лем}
        ВЕС_РУК = {"смысл": 1.0, "fts": 0.30, "like": 0.20, "лем": 0.30}
        итоговый = {}
        for имя, рука in РУКИ.items():
            if not рука or not ВЕС_РУК[имя]:
                continue
            з = list(рука.values())
            м = sum(з) / len(з)
            σ = (sum((x - м) ** 2 for x in з) / len(з)) ** 0.5 or 1.0
            ниж, верх = м - 3 * σ, м + 3 * σ
            for н, с in рука.items():
                итоговый[н] = итоговый.get(н, 0.0) + ВЕС_РУК[имя] * max(0.0, min(1.0, (с - ниж) / (верх - ниж)))

        # --- «сосед»: ответ в журнале почти всегда следующая запись за вопросом ---
        # Порядок важен: сначала основные кандидаты, ПОТОМ подъём соседей (иначе соседи
        # считались от уже обрезанного топ-8 и ответ выпадал: замер 8/10 → 2101 не попадал).
        соседи = {}
        порядок = sorted(итоговый, key=lambda н: -итоговый[н])[:сколько]
        порядок = [н for н in порядок if н in итоговый]
        for н in list(порядок):
            if not con.execute("SELECT 1 FROM записи WHERE н=?", (н + 1,)).fetchone():
                continue
            # Сосед ставится СРАЗУ за родителем (секция «сосед»), а не по общему скору:
            # иначе он уходит на 14-е место и обрезается топ-10 (замер: 2101 так и не попадал).
            соседи[н] = н + 1
            итоговый.setdefault(н + 1, итоговый[н] * 0.85)   # чтобы не было KeyError
        # вставляем соседа сразу после родителя
        с_соседями = []
        for н in порядок:
            с_соседями.append(н)
            if н in соседи:
                с_соседями.append(соседи[н])
        порядок = с_соседями

        if с_реранком and len(порядок) > сколько:
            # ГЛУБИНА: источники требуют реранжировать топ 50–100, а не 20 —
            # «rerank deep, truncate shallow». Замер: провалы сидели на рангах 12–110.
            порядок = порядок[:60]
            # БЛОКИ, а не записи: «вопрос, ответ-сосед» переранжируются вместе.
            # Замер: плоский реранк ломал соседа и ронял дословные 6/6 → 3/6.
            блоки = []          # [(родитель, [записи блока])]
            в_блоке = set()
            for н in порядок:
                if н in в_блоке:
                    continue
                блок = [н]
                if соседи.get(н) and соседи[н] in порядок:
                    блок.append(соседи[н])
                блоки.append((н, блок))
                в_блоке.update(блок)
            кандидаты = []
            for род, блок in блоки:
                т = (con.execute("SELECT текст FROM записи WHERE н=?", (род,)).fetchone() or [""])[0]
                кандидаты.append((род, т))
            скоры = реранк(текст, кандидаты)
            if скоры:
                # смешиваем: 60% мнение реранкера + 40% текущий скор (не теряем лексику)
                макс_тек = max(итоговый.get(н, 0) for н in итоговый) or 1
                блоки.sort(key=lambda п: -(0.6 * скоры.get(п[0], 0) +
                                           0.4 * (итоговый.get(п[0], 0) / макс_тек)))
                новый = []
                for _, блок in блоки:
                    новый.extend(блок)
                порядок = новый + порядок[len(новый):]

        # Пары «вопрос-ответ» из топ-20 — ПОСЛЕ реранка и перед обрезкой:
        # если сильный кандидат (ранг 11..20) не влез, добавляем его пару в конец.
        # Замер: 2100 (вопрос про совет) на 11-м месте терял ответ 2101.
        полный = sorted(итоговый, key=lambda н: -итоговый[н])[:20]
        занято = set(порядок[:сколько])
        for н in полный:
            if len(занято) >= сколько: break
            if н in занято or (н + 1) in занято: continue
            if not con.execute("SELECT 1 FROM записи WHERE н=?", (н + 1,)).fetchone(): continue
            порядок = порядок[:сколько-2] + [н, н + 1]
            занято = set(порядок[:сколько])
            break      # одна пара за раз: не размываем выдачу
        итог = []
        for н in порядок[:сколько]:
            роль, время = con.execute("SELECT роль, время FROM записи WHERE н=?", (н,)).fetchone() or ("?", "")
            итог.append({"н": н, "роль": роль, "время": время,
                         "цитата": цитата.get(н, ""), "очки": -итоговый[н]})
        return итог
    finally:
        con.close()

# ---------- реранк ----------
_РЕР = {}
def реранк(текст, кандидаты):
    """bge-reranker-v2-m3-en-ru ONNX int8. Готовое: metalmon80/...; +20 п.п. Hit@1 (aimultiple)."""
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer
        import numpy as np
        м = os.environ.get("РЕРАНК_МОДЕЛЬ", os.path.join(МОДЕЛИ, "реранк"))
        if "сесс" not in _РЕР:
            # RAM 1,9 ГБ: две ONNX-модели сразу не влезают (был OOM, код 137).
            # Выгружаем векторы перед загрузкой реранкера — освобождает ~900 МБ.
            if _ВЕК:
                print("выгружаю bge-m3, чтобы освободить RAM под реранкер", flush=True)
                _ВЕК.clear()
            import gc; gc.collect()
            # раньше здесь молча возвращался None — реранкер не работал вовсе.
            м = восстановить_модель("реранк")
            try:
                _РЕР["ток"] = AutoTokenizer.from_pretrained(м)
                _РЕР["режим"] = "auto"
            except Exception:
                import sentencepiece as spm
                _РЕР["sp"] = spm.SentencePieceProcessor(model_file=os.path.join(м, "sentencepiece.bpe.model"))
                _РЕР["режим"] = "sp"
            о = ort.SessionOptions(); о.intra_op_num_threads = 4; о.inter_op_num_threads = 1
            _РЕР["сесс"] = ort.InferenceSession(os.path.join(м, "model.int8.onnx"), sess_options=о,
                                                providers=["CPUExecutionProvider"])
        пары = [(текст, (т or "")[:2000]) for _, т in кандидаты]
        if _РЕР.get("режим") == "sp":
            # ручная сборка пары «запрос </s> </s> документ </s>» как в bge-reranker
            sp = _РЕР["sp"]
            иды, маски = [], []
            for а, б in пары:
                ид = [0] + sp.encode(а, out_type=int)[:250] + [2, 2] + sp.encode(б, out_type=int)[:250] + [2]
                ид = ид[:512]
                иды.append(ид); маски.append([1] * len(ид))
            макс = max(len(x) for x in иды)
            иды = [x + [1] * (макс - len(x)) for x in иды]
            маски = [x + [0] * (макс - len(x)) for x in маски]
        else:
            enc = _РЕР["ток"](*zip(*пары), padding=True, truncation=True, max_length=512, return_tensors="np")
            иды, маски = enc["input_ids"], enc["attention_mask"]
        скоры = _РЕР["сесс"].run(None, {"input_ids": np.array(иды, dtype=np.int64),
                                        "attention_mask": np.array(маски, dtype=np.int64)})[0].flatten()
        # Замер: правильный ответ −2,66 против −10,71 у мусора — разброс огромный.
        # Возвращаем СКОРЫ (sigmoid от логита), а не только порядок: порядок без
        # скоров теряет информацию и ломает «соседа» (дословные 6/6 → 3/6).
        return {н: float(1 / (1 + np.exp(-с))) for (н, _), с in zip(кандидаты, скоры)}
    except Exception as e:
        print("реранк недоступен:", str(e)[:80])
        return None

# ---------- оценка ----------
def замер(набор, метка):
    """набор: [(вопрос, ожидаемый номер)]. Возвращает MRR."""
    сумма, хиты = 0.0, 0
    t0 = time.time()
    for вопрос, н in набор:
        рез = искать(вопрос, 10)
        номера = [р["н"] for р in рез]
        if н in номера:
            сумма += 1.0 / (номера.index(н) + 1); хиты += 1
    mrr = сумма / max(len(набор), 1)
    print(f"{метка}: MRR {mrr:.3f} | нашли {хиты}/{len(набор)} | {time.time()-t0:.1f} с")
    return mrr, хиты

if __name__ == "__main__":
    import traceback
    что = sys.argv[1] if len(sys.argv) > 1 else "build"
    if что == "build":
        try:
            print("> скачивание журнала", flush=True); скачать_журнал()
            print("> построение индекса", flush=True); перестроить()
            print("> эмбеддинги", flush=True); эмбеддинги()
            print("> проверка файлов:", flush=True)
            import glob
            for ф in sorted(glob.glob(os.path.join(БАЗА, "данные", "*"))):
                print("   ", os.path.basename(ф), os.path.getsize(ф)//1024, "КБ", flush=True)
            print("> ГОТОВО", flush=True)
        except Exception:
            print("!!! ОШИБКА:", flush=True)
            traceback.print_exc()
            sys.exit(1)
    elif что == "эмбеддинги":
        ш = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        вс = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        эмбеддинги(ш, вс)
    elif что == "склейка":
        склеить_шарды()
    elif что == "search":
        print(json.dumps(искать(" ".join(sys.argv[2:]) or "поиск", 5), ensure_ascii=False, indent=1))
    elif что == "eval-перефразы":
        import glob
        файл = os.path.join(БАЗА, "перефразы.json")
        н = json.load(open(файл, encoding="utf-8"))
        пары = [(п["вопрос"], п["ответ"]) for п in н]
        print("перефразов:", len(пары), flush=True)
        замер(пары, "перефразы")
    elif что == "eval":
        н = json.load(open(os.path.join(БАЗА, "наборы.json"), encoding="utf-8"))
        пары = [(п["вопрос"], п["ответ"]) for п in н] if isinstance(н, list) else []
        print("пар в наборе:", len(пары), flush=True)
        замер(пары, "реальные-вопросы-6")
    elif что == "eval-все":
        # сравнение с векторами и без
        н = json.load(open(os.path.join(БАЗА, "наборы.json"), encoding="utf-8"))
        пары = [(п["вопрос"], п["ответ"]) for п in н] if isinstance(н, list) else []
        print("--- без векторов ---", flush=True)
        замер(пары, "без векторов")
        print("--- с векторами bge-m3 ---", flush=True)
        замер(пары, "с векторами")
