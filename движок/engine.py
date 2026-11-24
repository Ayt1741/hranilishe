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
            т = (з.get("текст") or з.get("с") or "").strip()
            if т:
                yield з.get("н"), з.get("роль", "?"), з.get("время", ""), т

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

def перестроить(тихо=False):
    t0 = time.time()
    if os.path.exists(ИНДЕКС):
        os.remove(ИНДЕКС)
    con = sqlite3.connect(ИНДЕКС)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(СХЕМА)
    н_записей = 0; пачка = []
    for н, роль, время, текст in записи():
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
def эмбеддинги():
    """bge-m3 на CPU. 16 ГБ RAM облака хватает (1,6 ГБ модель + 49к×1024×4 = 200 МБ векторы)."""
    import numpy as np
    if os.path.exists(ВЕКТ):
        print("эмбеддинги уже есть:", os.path.getsize(ВЕКТ)//1048576, "МБ")
        return ВЕКТ
    from sentence_transformers import SentenceTransformer
    con = sqlite3.connect(ИНДЕКС)
    стр = list(con.execute("SELECT ид, н, substr(текст,1,1200) FROM куски"))
    con.close()
    print("кусков для эмбеддингов:", len(стр), flush=True)
    if not стр:
        raise RuntimeError("нет кусков — индекс пуст")
    print("загружаю bge-m3...", flush=True)
    м = SentenceTransformer("BAAI/bge-m3", device="cpu")
    print("модель загружена, размерность:", м.get_sentence_embedding_dimension(), flush=True)
    тексты = [s[2] or "" for s in стр]
    части_E, B = [], 64
    t0 = time.time()
    for i in range(0, len(тексты), B):
        E = м.encode(тексты[i:i+B], batch_size=B, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False, max_length=512)
        части_E.append(E.astype(np.float32))
        if i % 3200 == 0:
            print(f"  {i}/{len(тексты)} {time.time()-t0:.0f} с", flush=True)
    E = np.vstack(части_E)
    os.makedirs(os.path.dirname(ВЕКТ), exist_ok=True)
    np.save(ВЕКТ.replace(".npy", "_ids.npy"), np.array([s[0] for s in стр], dtype=np.int64))
    np.save(ВЕКТ.replace(".npy", "_ном.npy"), np.array([s[1] for s in стр], dtype=np.int64))
    np.save(ВЕКТ, E)
    размер = os.path.getsize(ВЕКТ)//1048576
    print("эмбеддинги:", E.shape, размер, "МБ,", round(time.time()-t0), "с", flush=True)
    if not os.path.exists(ВЕКТ):
        raise RuntimeError("файл векторов не создан")
    return ВЕКТ

def векторный_поиск(текст, сколько=40):
    """Косинус по bge-m3. Возвращает [номер] по убыванию близости."""
    import numpy as np
    if not os.path.exists(ВЕКТ):
        return []
    from sentence_transformers import SentenceTransformer
    if "модель" not in _ВЕК:
        _ВЕК["модель"] = SentenceTransformer("BAAI/bge-m3", device="cpu")
        _ВЕК["E"] = np.load(ВЕКТ, mmap_mode="r")
        _ВЕК["ном"] = np.load(ВЕКТ.replace(".npy","_ном.npy"))
    q = _ВЕК["модель"].encode([текст], normalize_embeddings=True, convert_to_numpy=True)[0]
    скор = np.asarray(_ВЕК["E"] @ q)
    лучшие = np.argsort(-скор)[:сколько*3]
    вида, итог = set(), []
    for ин in лучшие:
        н = int(_ВЕК["ном"][ин])
        if н in вида: continue
        вида.add(н); итог.append(н)
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

def искать(текст, сколько=10, к=60, с_реранком=True):
    con = sqlite3.connect(ИНДЕКС)
    try:
        группы = группы_слов(текст, con)
        if not группы:
            return []
        счета, покрытие, цитата = {}, {}, {}
        for г in группы:
            for ранг, (н, кусок) in enumerate(con.execute("""
                    SELECT к.н, snippet(куски_fts, 0, '«', '»', ' … ', 18)
                    FROM куски_fts JOIN куски к ON к.ид = куски_fts.rowid
                    WHERE куски_fts MATCH ? ORDER BY rank LIMIT 400""", (г,)).fetchall()):
                if н in цитата and цитата[н][2] <= ранг:
                    continue
                счета[н] = счета.get(н, 0.0) + 1.0 / (ранг + к)
                покрытие[н] = покрытие.get(н, 0) + 1
                цитата[н] = (кусок, ранг, ранг)
        # LIKE-рука (подстроки без морфологии)
        слова_like = [с for с in re.findall(r"[а-яёa-z0-9]{2,}", (текст or "").lower()) if с not in СТОП][:6]
        for с in слова_like:
            for ранг, (н,) in enumerate(con.execute(
                    "SELECT DISTINCT н FROM куски WHERE текст_нижн LIKE ? LIMIT 400", (f"%{с}%",)).fetchall()):
                счета[н] = счета.get(н, 0.0) + 1.0 / (ранг + к)
                покрытие[н] = покрытие.get(н, 0) + 1
                if н not in цитата:
                    стр = con.execute("SELECT текст FROM записи WHERE н=?", (н,)).fetchone()
                    цитата[н] = ((стр[0] or "")[:300], ранг, 10**6)
        # леммы (вес 0.5)
        леммы = [лемма(с) for с in re.findall(r"[а-яёa-z0-9]{2,}", (текст or "").lower())]
        if леммы:
            условие = " AND ".join(["л.леммы LIKE ?"] * len(леммы))
            for ранг, (н,) in enumerate(con.execute(
                    f"SELECT л.н FROM записи_леммы л WHERE {условие} LIMIT 400",
                    tuple(f"%{с}%" for с in леммы)).fetchall()):
                счета[н] = счета.get(н, 0.0) + 0.5 / (ранг + к)
        # Векторная рука bge-m3 (облако): семантика ловит перефразы без общих слов.
        try:
            _в = векторный_поиск(текст, 30)
            for _ранг, _н in enumerate(_в):
                счета[_н] = счета.get(_н, 0.0) + 1.0 / (_ранг + к)
                if _н not in цитата:
                    _стр = con.execute("SELECT текст FROM записи WHERE н=?", (_н,)).fetchone()
                    цитата[_н] = ((_стр[0] if _стр else "")[:300], 10**6, 10**6)
        except Exception:
            pass
        надо = len(группы)
        полные = sorted([н for н in счета if покрытие.get(н, 0) >= надо], key=lambda н: -счета[н])
        if len(полные) < сколько:
            полные += sorted([н for н in счета if покрытие.get(н, 0) < надо], key=lambda н: -счета[н])
        if с_реранком and len(полные) > сколько:
            кандидаты = []
            for н in полные[:20]:
                т = (con.execute("SELECT текст FROM записи WHERE н=?", (н,)).fetchone() or [""])[0]
                кандидаты.append((н, т))
            порядок = реранк(текст, кандидаты)
            if порядок:
                полные = порядок + [н for н in полные[20:]]
        итог = []
        for н in полные[:сколько]:
            роль, время = con.execute("SELECT роль, время FROM записи WHERE н=?", (н,)).fetchone() or ("?", "")
            итог.append({"н": н, "роль": роль, "время": время,
                         "цитата": цитата.get(н, ("", 0, 0))[0], "очки": -счета.get(н, 0.0)})
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
        м = os.environ.get("РЕРАНК_МОДЕЛЬ", os.path.join(БАЗА, "модели", "реранк"))
        if "сесс" not in _РЕР:
            if not os.path.exists(os.path.join(м, "model.int8.onnx")):
                return None
            _РЕР["ток"] = AutoTokenizer.from_pretrained(м)
            о = ort.SessionOptions(); о.intra_op_num_threads = 4; о.inter_op_num_threads = 1
            _РЕР["сесс"] = ort.InferenceSession(os.path.join(м, "model.int8.onnx"), sess_options=о,
                                                providers=["CPUExecutionProvider"])
        пары = [(текст, (т or "")[:2000]) for _, т in кандидаты]
        enc = _РЕР["ток"](*zip(*пары), padding=True, truncation=True, max_length=512, return_tensors="np")
        скоры = _РЕР["сесс"].run(None, {k: enc[k] for k in enc})[0].flatten()
        return [н for (н, _), _с in sorted(zip(кандидаты, скоры), key=lambda x: -x[1])]
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
    elif что == "search":
        print(json.dumps(искать(" ".join(sys.argv[2:]) or "поиск", 5), ensure_ascii=False, indent=1))
    elif что == "eval":
        наборы = json.load(open(os.path.join(БАЗА, "наборы.json"), encoding="utf-8"))
        for имя, набор in наборы.items():
            замер([(a, b) for a, b in набор], имя)
