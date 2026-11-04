#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Агент флота на раннере GitHub Actions.
Работает в настоящем браузере с chat.deepseek.com в режиме Expert (= V4-Pro).

Что делает на каждом шарде (S01..S20):
 1) берёт свою задачу из tasks.json;
 2) делает свой черновик на бесплатной модели (OpenRouter :free);
 3) отправляет задачу в браузер DeepSeek Expert и получает ответ Pro;
 4) отправляет Pro же сравнение «мой черновик против твоего ответа» — Pro выносит вердикт;
 5) пишет всё в result-<TAG>.jsonl.

Переменные среды:
  TAG              — метка шарда (S01)
  TASKS_FILE       — путь к задачам (по умолчанию tasks.json)
  DEEPSEEK_COOKIE  — cookie входа в DeepSeek (как ARENA_COOKIE в старом воркере)
  OPENROUTER_KEY   — ключ для бесплатных моделей (:free)
  HEADLESS         — 1 = без окна (на раннере всегда 1)
"""
import asyncio
import json
import os
import pathlib
import sys
import time
import urllib.request

TAG = os.environ.get("TAG", "S00")
HEADLESS = os.environ.get("HEADLESS", "1") == "1"
COOKIE = os.environ.get("DEEPSEEK_COOKIE", "")
OR_KEY = os.environ.get("OPENROUTER_KEY", "")
АДРЕС = "https://chat.deepseek.com/"
СВОБОДНАЯ = "nvidia/nemotron-3-ultra-550b-a55b:free"
ВЫХОД = pathlib.Path("result-%s.jsonl" % TAG)
ПОЛЯ = ["textarea", "div[contenteditable='true']", "[role='textbox']"]
ОТВЕТЫ = ["div[class*='ds-markdown']", ".ds-markdown", "div[class*='markdown']"]


# --------------------------- своя модель (бесплатно) ---------------------------
def свой_черновик(задача):
    if not OR_KEY:
        return "", "нет ключа OpenRouter"
    тело = json.dumps({"model": СВОБОДНАЯ,
                       "messages": [{"role": "user", "content": задача}],
                       "max_tokens": 900}).encode()
    r = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=тело,
                               headers={"Content-Type": "application/json",
                                        "Authorization": "Bearer " + OR_KEY,
                                        "User-Agent": "FleetBrowser/1.0"})
    try:
        d = json.load(urllib.request.urlopen(r, timeout=180))
        return (d["choices"][0]["message"].get("content") or "").strip(), "ок"
    except Exception as е:
        return "", "ошибка: %s" % str(е)[:80]


# ------------------------------- браузер -------------------------------------
async def первый(стр, селекторы, таймаут=45000):
    конец = time.time() + таймаут / 1000
    while time.time() < конец:
        for с in селекторы:
            try:
                э = стр.locator(с).first
                if await э.count() and await э.is_visible():
                    return э
            except Exception:
                continue
        await asyncio.sleep(0.5)
    return None


async def expert(стр):
    for т in ["Expert", "Expert Mode", "DeepThink"]:
        try:
            к = стр.get_by_text(т, exact=False).first
            if await к.count():
                await к.click(timeout=2500)
                await asyncio.sleep(0.8)
                return т
        except Exception:
            continue
    return "не найден"


async def спросить(стр, вопрос, ждать=240):
    поле = await первый(стр, ПОЛЯ)
    if not поле:
        return "", "поле ввода не найдено"
    try:
        await поле.click()
        await поле.fill(вопрос)
    except Exception:
        await поле.type(вопрос)
    await asyncio.sleep(0.3)
    await стр.keyboard.press("Enter")
    прошлый, стабильно, t0 = "", 0, time.time()
    while time.time() - t0 < ждать:
        await asyncio.sleep(2)
        текущий = ""
        for с in ОТВЕТЫ:
            try:
                б = стр.locator(с)
                if await б.count():
                    текущий = await б.last.inner_text()
                    break
            except Exception:
                continue
        if текущий and текущий == прошлый:
            стабильно += 1
            if стабильно >= 3:
                return текущий, "ок %.0f с" % (time.time() - t0)
        else:
            стабильно = 0
        прошлый = текущий
    return прошлый, "таймаут %d с" % ждать


async def через_браузер(задачи):
    from playwright.async_api import async_playwright
    итоги = []
    async with async_playwright() as п:
        бр = await п.chromium.launch(headless=HEADLESS,
                                     args=["--no-sandbox", "--disable-dev-shm-usage",
                                           "--disable-blink-features=AutomationControlled"])
        контекст = await бр.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36")
        if COOKIE:
            печенье = []
            for кусок in COOKIE.split(";"):
                if "=" in кусок:
                    имя, знач = кусок.split("=", 1)
                    печенье.append({"name": имя.strip(), "value": знач.strip(),
                                    "domain": ".deepseek.com", "path": "/"})
            if печенье:
                await контекст.add_cookies(печенье)
        стр = await контекст.new_page()
        await стр.goto(АДРЕС, timeout=90000, wait_until="domcontentloaded")
        await asyncio.sleep(6)
        код = (await стр.inner_text("body"))[:200].replace("\n", " ")
        режим = await expert(стр)
        for задача, черновик in задачи:
            ответ, статус = await спросить(стр, задача)
            вердикт = ""
            if ответ:
                сравнение = ("Сравни два ответа на задачу и вынеси вердикт.\nЗАДАЧА: %s\n\n"
                             "ОТВЕТ А (другая модель):\n%s\n\nОТВЕТ Б (твой):\n%s\n\n"
                             "Напиши: что верно у А, что верно у Б, где ошибка, и итоговый лучший ответ." %
                             (задача, черновик[:1500], ответ[:2500]))
                вердикт, _ = await спросить(стр, сравнение)
            итоги.append({"задача": задача, "черновик": черновик, "ответ_про": ответ,
                          "сравнение": вердикт, "статус": статус, "режим": режим, "страница": код})
        await контекст.close()
        await бр.close()
    return итоги


def main():
    файл = os.environ.get("TASKS_FILE", "tasks.json")
    путь = pathlib.Path(файл)
    задачи_все = json.loads(путь.read_text(encoding="utf-8")) if путь.exists() else [
        {"текст": "Назови 3 способа ускорить скачивание файла при слабом интернете."}]
    # свой шард берёт свою задачу (номер шарда = номер задачи по кругу)
    номер = int(TAG[1:]) if TAG[1:].isdigit() else 0
    моя = задачи_все[номер % len(задачи_все)]["текст"] if задачи_все else ""
    print("шард %s · задача: %s" % (TAG, моя[:120]))
    черновик, ст = свой_черновик(моя)
    print("свой черновик: %d символов (%s)" % (len(черновик), ст))
    итоги = asyncio.run(через_браузер([(моя, черновик)]))
    с = итоги[0] if итоги else {}
    строка = {"шард": TAG, "время": int(time.time()), "задача": моя,
              "своих_символов": len(черновик), "про_символов": len(с.get("ответ_про", "")),
              "сравнение_символов": len(с.get("сравнение", "")),
              "статус": с.get("статус"), "режим": с.get("режим"),
              "страница": (с.get("страница") or "")[:200]}
    with ВЫХОД.open("a", encoding="utf-8") as ж:
        ж.write(json.dumps(строка, ensure_ascii=False) + "\n")
    print(json.dumps(строка, ensure_ascii=False, indent=1))
    if not с.get("ответ_про"):
        print("ПРО НЕ ОТВЕТИЛ — смотри поле «страница», там что вернул сервер")
        sys.exit(3)


if __name__ == "__main__":
    main()
