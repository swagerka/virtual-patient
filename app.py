import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
import random
import re
import json
from datetime import datetime
import time # Для таймера
import pandas as pd # Для истории сессий

# --- Константы ---
MEDICAL_SPECIALIZATIONS = ["Общая терапия", "Гастроэнтерология", "Кардиология", "Пульмонология", "Неврология", "Эндокринология", "Нефрология (Урология)", "Инфекционные болезни", "Ревматология", "Педиатрия (общие случаи)", "Травматология и Ортопедия (несложные случаи)", "Гинекология (базовые случаи)", "Дерматология", "Психиатрия (базовые случаи)", "Офтальмология (базовые случаи)", "ЛОР (базовые случаи)"]
AGE_RANGES = {"Младенец/Ребенок (0-5 лет)": (0, 5), "Ребенок (6-12 лет)": (6, 12), "Подросток (13-17 лет)": (13, 17), "Молодой взрослый (18-35 лет)": (18, 35), "Средний возраст (36-60 лет)": (36, 60), "Пожилой (61-80 лет)": (61, 80), "Старческий (81+ лет)": (81, 100)}
GENDERS = ["Любой", "Мужской", "Женский"]
DIFFICULTY_LEVELS = ["Легкий", "Средний", "Тяжелый", "Экспертный"]

DIFFICULTY_LEVELS_DETAILS = {
    "Легкий": {
        "description": "Симптомы КЛАССИЧЕСКИЕ, один ведущий синдром. Пациент КОНТАКТНЫЙ, ЯСНО излагает, предоставляет всю информацию легко. Результаты физикального осмотра ОДНОЗНАЧНЫ. Диф. диагноз МИНИМАЛЕН. Сопутствующие заболевания ОТСУТСТВУЮТ или не влияют.",
        "patient_persona_modifier": "Ты очень кооперативный пациент, всегда готов помочь врачу, ясно и полно отвечаешь на все вопросы. Твое состояние не вызывает у тебя сильного беспокойства."
    },
    "Средний": {
        "description": "Симптомы СЛЕГКА АТИПИЧНЫЕ или 2-3 симптома. Возможен один 'красный флаг'. Пациент может быть НЕМНОГО ВСТРЕВОЖЕН, давать информацию НЕ СРАЗУ, а после уточняющих вопросов. Результаты осмотра могут требовать ИНТЕРПРЕТАЦИИ. Диф. диагноз с 1-2 состояниями. 1-2 НЕКРИТИЧНЫХ сопутствующих заболевания.",
        "patient_persona_modifier": "Ты пациент, который немного обеспокоен своим состоянием. Ты стараешься отвечать на вопросы, но иногда можешь что-то упустить, если врач не спросит прямо. Иногда можешь переспросить или выразить легкое волнение."
    },
    "Тяжелый": {
        "description": "Симптомы АТИПИЧНЫЕ, МНОЖЕСТВЕННЫЕ, МАСКИРУЮЩИЕСЯ. Несколько 'красных флагов'. Пациент может быть ТРУДНЫМ В ОБЩЕНИИ (скрытным, раздражительным, многословным и уводящим от темы). Может давать ПРОТИВОРЕЧИВУЮ информацию. Результаты осмотра НЕОДНОЗНАЧНЫ. Обширный ДИФ. ДИАГНОЗ. ЗНАЧИМЫЕ сопутствующие заболевания. Возможны 'ЛОЖНЫЕ СЛЕДЫ'.",
        "patient_persona_modifier": "Ты сложный пациент. Возможно, ты напуган, раздражен или не доверяешь врачам. Ты можешь быть немногословен, или наоборот, говорить слишком много о несущественных деталях. Врачу придется постараться, чтобы получить от тебя нужную информацию. Твои ответы могут быть не всегда прямыми или полными с первого раза."
    },
    "Экспертный": {
        "description": "Все признаки ТЯЖЕЛОГО + РЕДКИЕ заболевания, СЛОЖНЫЕ ЭТИЧЕСКИЕ ДИЛЕММЫ, необходимость сообщить плохие новости. Пациент с ВЫРАЖЕННЫМИ КОММУНИКАТИВНЫМИ БАРЬЕРАМИ (языковой, психологический). Возможна необходимость принятия решений в условиях ОГРАНИЧЕННЫХ РЕСУРСОВ (симулируется).",
        "patient_persona_modifier": "Ты очень сложный пациент. У тебя могут быть серьезные психологические проблемы, языковой барьер, или ты можешь быть настроен крайне скептически или враждебно. Возможно, тебе нужно сообщить очень плохие новости, и твоя реакция будет сильной. Твои симптомы могут быть крайне запутанными и указывать на очень редкое заболевание."
    }
}
TIMER_DURATIONS_MINUTES = [10, 15, 20, 25, 30, 40, 60]
MAX_CONSULTATIONS = 3

# --- Загрузка сценариев ---
try:
    from scenarios_data import SCENARIOS
    if not isinstance(SCENARIOS, list):
        st.warning("Переменная SCENARIOS в scenarios_data.py не является списком. Используется пустой список.")
        SCENARIOS = []
except ImportError:
    st.warning("Файл scenarios_data.py не найден. Будут доступны только LLM-сгенерированные сценарии.")
    SCENARIOS = []
except Exception as e:
    st.error(f"Произошла ошибка при загрузке scenarios_data.py: {e}")
    SCENARIOS = []

# --- Конфигурация и API клиент ---
load_dotenv()
KOBOLD_API_URL = os.getenv("KOBOLD_API_URL", "http://localhost:5002/v1/")
LLM_TIMEOUT_SHORT = 60.0  # seconds for quick responses
LLM_TIMEOUT_LONG = 300.0 # seconds for long generation/evaluation

try:
    client = OpenAI(base_url=KOBOLD_API_URL, api_key="sk-not-needed")
except Exception as e:
    st.error(f"Ошибка инициализации OpenAI клиента: {e}. Убедитесь, что KoboldCpp или совместимый LLM сервер запущен и доступен по адресу {KOBOLD_API_URL}.")
    st.stop()

# --- Вспомогательные функции для парсинга JSON ---
def _extract_and_parse_json(raw_text):
    if not raw_text or not raw_text.strip():
        raise ValueError("Ответ LLM пуст или отсутствует.")
    json_text = None
    json_block_match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
    if json_block_match:
        json_text = json_block_match.group(1).strip()
    else:
        json_start_index = raw_text.find('{')
        json_end_index = raw_text.rfind('}')
        if json_start_index != -1 and json_end_index != -1 and json_end_index > json_start_index:
            json_text = raw_text[json_start_index : json_end_index + 1].strip()
    
    if not json_text:
        error_detail = []
        if not json_block_match: error_detail.append("JSON блок ```json ... ``` не найден")
        json_braces_check_start = raw_text.find('{')
        json_braces_check_end = raw_text.rfind('}')
        if not (json_braces_check_start != -1 and json_braces_check_end != -1 and json_braces_check_end > json_braces_check_start):
            error_detail.append("валидная структура '{...}' не найдена")
        if not error_detail: error_detail.append("неизвестная причина отсутствия извлекаемой JSON структуры")
        raise ValueError(f"JSON структура не найдена в ответе LLM. Детали: {', и '.join(error_detail)}.")

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        st.warning(f"Ошибка парсинга извлеченного JSON ('{json_text[:100].strip().replace(chr(10), ' ')}...'): {e.msg}. Попытка восстановления...")
        
        # Ремонт A: Исправление неправильно закавыченных ключей (например, key": -> "key":)
        try:
            # (?<=[{,]\s*) - lookbehind for '{' or ',' followed by optional whitespace (problematic due to \s*)
            # ([a-zA-Z_][a-zA-Z0-9_]*) - key name (group 1)
            # " - misplaced quote
            # (\s*:) - whitespace and colon (group 2)
            # Replacement: "\1"\2 -> "key_name":
            # Fixed regex to avoid variable-width lookbehind:
            repaired_text_A = re.sub(r'([{,])(\s*)([a-zA-Z_][a-zA-Z0-9_]*)"(\s*:)', r'\1\2"\3"\4', json_text)
            if repaired_text_A != json_text: 
                st.info("Ремонт A: Попытка исправить ключи с неправильными кавычками (например, 'key\":' -> '\"key\":')...")
                return json.loads(repaired_text_A) 
        except json.JSONDecodeError as e_fixA:
            st.warning(f"Ремонт A (исправление ключей) не удался или привел к новой ошибке парсинга: {e_fixA.msg}")
        except Exception as repair_eA: # Catch regex errors like "look-behind requires fixed-width pattern" if any remain
            st.warning(f"Неожиданная ошибка во время Ремонта A (исправление ключей): {repair_eA}")
        
        # Ремонт B: Поиск полного JSON объекта путем балансировки скобок
        try:
            open_braces = 0; valid_json_end = -1; first_brace = json_text.find('{')
            if first_brace == -1: 
                raise ValueError("JSON-подобная структура (фигурная скобка '{') не найдена для попытки балансировки.")

            for i, char in enumerate(json_text[first_brace:]):
                actual_index = i + first_brace
                if char == '{': open_braces += 1
                elif char == '}': open_braces -= 1
                if open_braces == 0 and i > 0: 
                    potential_json = json_text[first_brace : actual_index + 1]
                    try:
                        json.loads(potential_json) 
                        valid_json_end = actual_index
                    except json.JSONDecodeError:
                        continue # Keep searching if this substring is not valid JSON itself
            
            if valid_json_end != -1:
                repaired_json_text_B = json_text[first_brace : valid_json_end + 1]
                st.info(f"Ремонт B: Восстановлено через поиск полного объекта: '{repaired_json_text_B[:100].strip().replace(chr(10), ' ')}...'")
                return json.loads(repaired_json_text_B)
        except Exception as repair_B_e: 
            st.warning(f"Ремонт B (поиск полного объекта) не удался: {repair_B_e}")

        # Ремонт C: Попытка добавить недостающие закрывающие символы
        possible_suffixes = ["}", "]", "}}", "]}", "}]", "\"}", "\"]}", "\"}]", "\"}}", ")}"]; possible_suffixes = list(dict.fromkeys(possible_suffixes)) 
        for suffix_to_add in possible_suffixes:
            try: 
                st.info(f"Ремонт C: Попытка добавить суффикс '{suffix_to_add}'...")
                return json.loads(json_text + suffix_to_add)
            except json.JSONDecodeError: 
                continue
        
        # Ремонт D: Попытка исправить отсутствующий ключ 'description' в объектах списка 'common_mistakes'
        # Пример ошибки: { "id": "some_id", "This is a description string.", "penalty": 3 }
        # Должно быть:   { "id": "some_id", "description": "This is a description string.", "penalty": 3 }
        try:
            # (\{\s*"id"\s*:\s*"[^"]*"\s*,\s*) - Group 1: Matches '{ "id": "value", '
            # ("[^\"]*")                        - Group 2: Matches the string value that's missing its key, e.g., "description text"
            # (\s*,\s*"penalty"\s*:)           - Group 3: Matches ' , "penalty": '
            # This regex assumes standard string content without internal escaped quotes for the description for simplicity.
            pattern_common_mistake_fix = r'(\{\s*"id"\s*:\s*"[^"]*"\s*,\s*)("[^"]*")(\s*,\s*"penalty"\s*:)'
            
            # Check if the pattern to fix exists to avoid unnecessary operations
            if re.search(pattern_common_mistake_fix, json_text):
                repaired_text_D = re.sub(pattern_common_mistake_fix,
                                         r'\1"description": \2\3', 
                                         json_text)
                if repaired_text_D != json_text: # Ensure a change was made
                    st.info("Ремонт D: Попытка исправить отсутствующий ключ 'description' в 'common_mistakes'...")
                    return json.loads(repaired_text_D)
        except json.JSONDecodeError as e_fixD:
            st.warning(f"Ремонт D (исправление common_mistakes) не удался или привел к новой ошибке парсинга: {e_fixD.msg}")
        except Exception as repair_eD:
            st.warning(f"Неожиданная ошибка во время Ремонта D (исправление common_mistakes): {repair_eD}")


        original_error_doc = e.doc if hasattr(e, 'doc') and e.doc == json_text else json_text
        original_error_pos = e.pos if hasattr(e, 'pos') else 0
        raise json.JSONDecodeError(f"Не удалось восстановить JSON ('{json_text[:100].strip().replace(chr(10), ' ')}...') после нескольких попыток. Исходная ошибка: {e.msg}",
            doc=original_error_doc, pos=original_error_pos)


# --- Функции ---
def get_time_remaining_str():
    if st.session_state.get("timer_active_in_scenario") and "time_remaining" in st.session_state:
        remaining = st.session_state.time_remaining
        if remaining is not None:
            minutes = int(remaining // 60); seconds = int(remaining % 60)
            return f"{minutes:02d}:{seconds:02d}"
    return "N/A"

def generate_llm_response(messages_history_for_llm, system_prompt_for_llm):
    try:
        messages_to_send = [{"role": "system", "content": system_prompt_for_llm}] + \
                           [msg for msg in messages_history_for_llm if msg["role"] in ["user", "assistant"]]
        response = client.chat.completions.create(
            model="local-model", 
            messages=messages_to_send, 
            max_tokens=450, 
            temperature=0.75,
            timeout=LLM_TIMEOUT_SHORT 
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        st.warning("LLM не вернул текстовый ответ для пациента."); return "Пациент задумался и молчит..."
    except Exception as e:
        st.error(f"Ошибка API при ответе пациента: {e}"); return "Возникла техническая проблема с пациентом, он не может сейчас ответить."

def get_consultant_response(patient_dialogue_history, user_question_to_consultant, specialist_type, main_scenario_info):
    system_prompt = f"""Ты — опытный врач-консультант, специалист в области {specialist_type}.
К тебе обратился коллега за советом по клиническому случаю.
Он предоставит тебе краткую информацию о пациенте, историю общения с пациентом и свой конкретный вопрос.
Твоя задача — дать краткий, но емкий и полезный совет, основанный на представленной информации.
Сосредоточься на ответе на вопрос коллеги. Не повторяй всю информацию, которую он тебе дал.
Помни, что окончательное решение принимает лечащий врач. Будь профессионален и лаконичен.
"""
    dialogue_history_str = "Диалог с пациентом:\n"
    for i, m in enumerate(patient_dialogue_history):
        role_translated = "Врач (коллега)" if m['role'] == 'user' else "Пациент"
        dialogue_history_str += f"Ход {i+1} {role_translated}: {m['content']}\n"

    user_content = f"""Информация о пациенте: {main_scenario_info}

{dialogue_history_str}

Вопрос от коллеги: {user_question_to_consultant}

Твой совет:
"""
    try:
        response = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=500,
            temperature=0.5,
            timeout=LLM_TIMEOUT_SHORT
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        return "Консультант не смог предоставить ответ."
    except Exception as e:
        st.error(f"Ошибка API при запросе консультации: {e}")
        return "Техническая проблема с консультантом. Попробуйте позже."

def generate_new_scenario_via_llm(age_range_str=None, specialization_str=None, gender_str=None, difficulty_str=None):
    st.info("Запрос на генерацию нового сценария отправлен LLM. Это может занять некоторое время...")
    customization_prompt_parts = []
    if age_range_str and age_range_str != "Любой" and age_range_str in AGE_RANGES:
        min_a, max_a = AGE_RANGES[age_range_str]; customization_prompt_parts.append(f"Возраст пациента: от {min_a} до {max_a} лет.")
    elif age_range_str == "Любой": customization_prompt_parts.append("Возраст пациента: любой.")
    if gender_str and gender_str != "Любой": customization_prompt_parts.append(f"Пол пациента: {gender_str}.")
    elif gender_str == "Любой": customization_prompt_parts.append("Пол пациента: любой.")

    actual_difficulty_str = difficulty_str if difficulty_str and difficulty_str in DIFFICULTY_LEVELS_DETAILS else "Средний"
    difficulty_details = DIFFICULTY_LEVELS_DETAILS[actual_difficulty_str]
    patient_persona_modifier_prompt = difficulty_details['patient_persona_modifier']
    customization_prompt_parts.append(f"Уровень сложности: {actual_difficulty_str}. {difficulty_details['description']}")

    actual_specialization_str = specialization_str
    if not specialization_str or specialization_str == "Любая": actual_specialization_str = random.choice(MEDICAL_SPECIALIZATIONS)
    customization_prompt_parts.append(f"Медицинская область: {actual_specialization_str}.")

    customization_instructions_str = f"ОСОБЫЕ ТРЕБОВАНИЯ К СЦЕНАРИЮ: {' '.join(customization_prompt_parts)}"
    system_prompt_for_generator = "Ты — эксперт по созданию детализированных медицинских обучающих симуляций. Твоя задача — сгенерировать полный сценарий для платформы 'Виртуальный Пациент'. Предоставь ответ СТРОГО в формате JSON. Не добавляй никакого другого текста до или после JSON. Убедись, что JSON валиден."
    user_prompt_for_generator = f"""
{customization_instructions_str}
Сценарий должен быть реалистичным и клинически правдоподобным. JSON должен иметь следующую структуру:
{{
  "id": "llm_gen_YYYYMMDDHHMMSS",
  "name": "str: Краткое, интригующее название сценария (например, 'Пациент с болью в груди', 'Загадочный случай кашля'). НАЗВАНИЕ НЕ ДОЛЖНО СОДЕРЖАТЬ ИЛИ НАМЕКАТЬ НА ДИАГНОЗ.",
  "difficulty_level_tag": "{actual_difficulty_str}",
  "patient_initial_info_display": "str: Инфо для врача (возраст, пол, краткая основная жалоба). Соответствует параметрам.",
  "patient_llm_persona_system_prompt": "str: Подробный системный промпт для LLM-пациента, описывающий его личность, предысторию болезни, манеру общения, детали текущего состояния, но НЕ включая общие инструкции по обработке команд осмотра или анализов (они будут добавлены приложением).
    - Этот промпт должен фокусироваться на уникальных аспектах пациента в данном сценарии.
    - Например: 'Ты 45-летний мужчина, работаешь строителем. Последние 3 дня тебя беспокоит кашель...'
    - ДОБАВЬ СЮДА ЭТУ ИНСТРУКЦИЮ ДЛЯ СЕБЯ (LLM-ГЕНЕРАТОРА): '{patient_persona_modifier_prompt}' в описание личности пациента.",
  "initial_patient_greeting": "str: Первая фраза пациента.",
  "true_diagnosis_internal": "str: Краткий истинный диагноз (например, 'Острый бронхит').",
  "true_diagnosis_detailed": "str: Подробное описание истинного диагноза, включая патогенез, ключевые клинические признаки и критерии диагностики.",
  "key_anamnesis_points": ["list", "of", "str: 5-7 КЛЮЧЕВЫХ моментов, которые врач должен выяснить из анамнеза для постановки диагноза."],
  "correct_plan_detailed": "str: ПОЛНЫЙ правильный план обследования и лечения, соответствующий диагнозу и стандартам.",
  "common_mistakes": [
    {{ "id": "empty_dx", "description": "Диагноз не был поставлен.", "penalty": 5 }},
    {{ "id": "empty_plan", "description": "План не был предложен.", "penalty": 5 }},
    {{ "id": "custom_mistake_1", "description": "str: Частая ошибка, специфичная для этого сценария (например, 'Недооценка красного флага X').", "penalty": "int (1-3)"}},
    {{ "id": "custom_mistake_2", "description": "str: Другая частая ошибка для этого сценария.", "penalty": "int (1-3)"}}
  ],
  "key_diagnostic_questions_keywords": ["list", "of", "str: 3-5 групп ключевых слов или фраз, относящихся к важным вопросам для сбора анамнеза (например, 'характер боли', 'когда началось')."],
  "correct_diagnosis_keywords_for_check": ["list", "of", "str: 2-4 ключевых слова из истинного диагноза для автоматической проверки (например, 'бронхит', 'острый')."],
  "correct_plan_keywords_for_check": ["list", "of", "str: 3-5 ключевых слов из правильного плана для автоматической проверки (например, 'оак', 'рентген', 'антибиотик')."],
  "available_investigations": {{
    "OAK": {{ "request_keywords": ["оак", "общий анализ крови", "клинический анализ крови"], "results_text": "Гемоглобин: 130 г/л, Эритроциты: 4.5х10^12/л, Лейкоциты: 9.5х10^9/л (палочкоядерные 8%, сегментоядерные 60%, лимфоциты 25%, моноциты 7%), СОЭ: 15 мм/ч", "turn_to_provide_results": 1 }},
    "Биохимия крови (базовая)": {{ "request_keywords": ["биохимия", "бх", "биохимический анализ"], "results_text": "Глюкоза: 5.0 ммоль/л, Креатинин: 80 мкмоль/л, Мочевина: 5.5 ммоль/л, Общий билирубин: 15 мкмоль/л, АЛТ: 25 Ед/л, АСТ: 30 Ед/л", "turn_to_provide_results": 2 }}
    /* Добавь 1-3 других РЕЛЕВАНТНЫХ для сценария исследования (например, ОАМ, ЭКГ, Рентген ОГК, УЗИ ОБП и т.д.) с их ключевыми словами для запроса, текстом результатов и задержкой предоставления. */
  }},
  "physical_exam_findings_prompt_details": {{
    "temperature": "37.2°C", "blood_pressure": "120/80 мм рт.ст.", "pulse": "78 уд/мин", "respiratory_rate": "16 в мин", "spo2": "98%",
    "general_condition": "удовлетворительное", "skin_mucous_membranes": "обычной окраски, чистые, влажные",
    "auscultation_lungs": "дыхание везикулярное, хрипов нет", "palpation_abdomen": "живот мягкий, безболезненный",
    "throat_inspection": "зев спокоен, миндалины не увеличены"
    /* Добавь или измени 2-4 других РЕЛЕВАНТНЫХ для сценария параметра физикального осмотра и их значения, которые пациент сообщит при соответствующем действии врача. Убедись, что есть данные для стандартных быстрых действий: температура, АД, SpO2, легкие, живот, горло. Также добавь поля для: 'skin_appearance', 'lymph_nodes', 'thyroid_palpation', 'joints_inspection', 'neuro_status_brief', 'liver_palpation', 'spleen_palpation', 'edema_check', 'peripheral_pulses', 'ear_inspection', 'nose_inspection', 'heart_rate' (может быть синонимом pulse). */
  }},
  "expected_differential_diagnoses": ["list", "of", "str: 2-3 наиболее вероятных дифференциальных диагноза для данного случая."],
  "communication_focus_points": ["list", "of", "str: (Для уровня 'Экспертный' или 'Тяжелый') 1-2 специфические коммуникативные задачи или вызова (например, 'Сообщение плохих новостей', 'Работа с недоверчивым пациентом'). Для 'Легкий' и 'Средний' оставь пустым."],
  "dynamic_state_triggers": [
    /* Пример: {{ "condition_type": "missed_key_question", "key_question_keyword": "аллергия", "turns_to_trigger": 5, "patient_response_cue": "Кстати, доктор, я вспомнил, у меня же аллергия на пенициллин!" }} */
    /* Добавь 0-1 динамический триггер, если это уместно для сценария. */
  ]
}}
Убедись, что ВСЕ поля JSON заполнены правдоподобной и клинически релевантной информацией. `patient_llm_persona_system_prompt` должен быть достаточно подробным, чтобы передать характер пациента и его историю, но БЕЗ общих инструкций по симуляции, так как они добавляются отдельно. Поле `name` НЕ должно раскрывать диагноз.
"""
    raw_text = ""
    try:
        messages_for_scenario_gen = [{"role": "system", "content": system_prompt_for_generator}, {"role": "user", "content": user_prompt_for_generator}]
        response = client.chat.completions.create(
            model="local-model", 
            messages=messages_for_scenario_gen, 
            max_tokens=8192, 
            temperature=0.7,
            timeout=LLM_TIMEOUT_LONG
        )
        if not (response.choices and response.choices[0].message and response.choices[0].message.content):
            st.error("LLM не вернул контент для генерации сценария."); return None
        raw_text = response.choices[0].message.content.strip()
        generated_scenario = _extract_and_parse_json(raw_text)
        generated_scenario['id'] = f"llm_gen_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        required_keys_with_defaults = {
            "id": None, "name": "Сгенерированный сценарий (название не указано)", "difficulty_level_tag": actual_difficulty_str,
            "patient_initial_info_display": "Информация о пациенте отсутствует.",
            "patient_llm_persona_system_prompt": "Вы пациент. Опишите свои жалобы. Ваше поведение и ответы должны соответствовать заданному уровню сложности.",
            "initial_patient_greeting": "Здравствуйте, доктор.", "true_diagnosis_internal": "Диагноз не указан.", "true_diagnosis_detailed": "Подробное описание диагноза отсутствует.",
            "key_anamnesis_points": [], "correct_plan_detailed": "Описание правильного плана отсутствует.",
            "common_mistakes": [{"id": "empty_dx", "description": "Диагноз не был поставлен.", "penalty": 5}, {"id": "empty_plan", "description": "План не был предложен.", "penalty": 5}],
            "key_diagnostic_questions_keywords": [], "correct_diagnosis_keywords_for_check": [], "correct_plan_keywords_for_check": [],
            "available_investigations": {},
            "physical_exam_findings_prompt_details": { 
                "temperature": "36.6°C", "blood_pressure": "120/80 мм рт.ст.", "pulse": "70 уд/мин", "spo2": "98%",
                "auscultation_lungs": "дыхание везикулярное, хрипов нет", "palpation_abdomen": "живот мягкий, безболезненный",
                "throat_inspection": "зев спокоен, налетов нет",
                "skin_appearance": "Кожные покровы обычной окраски и влажности, высыпаний нет.",
                "lymph_nodes": "Периферические лимфоузлы не увеличены, безболезненны.",
                "thyroid_palpation": "Щитовидная железа не увеличена, мягко-эластической консистенции, безболезненна.",
                "joints_inspection": "Суставы внешне не изменены, движения в полном объеме.",
                "neuro_status_brief": "Сознание ясное, ориентирован. Зрачки D=S, фотореакция живая. Речь внятная.",
                "liver_palpation": "Край печени по краю реберной дуги, безболезненный.",
                "spleen_palpation": "Селезенка не пальпируется.",
                "edema_check": "Отеков нет.",
                "peripheral_pulses": "Пульсация на периферических артериях удовлетворительная, симметричная.",
                "ear_inspection": "Наружные слуховые проходы свободны, барабанные перепонки серые, опознавательные знаки четкие.",
                "nose_inspection": "Слизистая носа розовая, влажная, носовые ходы свободны.",
                "heart_rate": "ЧСС 70 уд/мин, ритмичный." 
            },
            "expected_differential_diagnoses": [],
            "communication_focus_points": [], "dynamic_state_triggers": []
        }
        for key, default_value in required_keys_with_defaults.items():
            is_list_default, is_dict_default, is_str_default = isinstance(default_value, list), isinstance(default_value, dict), isinstance(default_value, str)
            if key not in generated_scenario:
                if default_value is not None: generated_scenario[key] = default_value; st.warning(f"Поле '{key}' отсутствовало в генерации LLM, установлено значение по умолчанию.")
                elif key != "id": st.error(f"Критическое поле '{key}' отсутствует в генерации LLM. Сценарий не может быть использован."); return None
            current_val = generated_scenario.get(key)
            if is_list_default and not isinstance(current_val, list): generated_scenario[key] = default_value; st.warning(f"Поле '{key}' должно быть списком, исправлено на значение по умолчанию.")
            elif is_dict_default and not isinstance(current_val, dict): generated_scenario[key] = default_value; st.warning(f"Поле '{key}' должно быть словарем, исправлено на значение по умолчанию.")
            elif is_str_default and not isinstance(current_val, str): generated_scenario[key] = str(current_val); st.warning(f"Поле '{key}' должно быть строкой, преобразовано в строку.")

        llm_common_mistakes = generated_scenario.get("common_mistakes", []); final_common_mistakes = []; ids_added = set()
        default_mistakes_map = {"empty_dx": {"description": "Диагноз не был поставлен.", "penalty": 5}, "empty_plan": {"description": "План не был предложен.", "penalty": 5}}
        for m_id_default, m_data_default in default_mistakes_map.items():
            found_in_llm = any(isinstance(m, dict) and m.get("id") == m_id_default for m in llm_common_mistakes)
            if found_in_llm:
                m_llm = next(m for m in llm_common_mistakes if isinstance(m, dict) and m.get("id") == m_id_default)
                final_common_mistakes.append({"id": m_id_default, "description": m_llm.get("description", m_data_default["description"]), "penalty": int(m_llm.get("penalty", m_data_default["penalty"]))})
            else: final_common_mistakes.append({"id": m_id_default, **m_data_default})
            ids_added.add(m_id_default)

        for mistake in llm_common_mistakes:
            if isinstance(mistake, dict) and mistake.get("id") and mistake.get("id") not in ids_added:
                try: penalty_val = int(mistake.get("penalty", 1))
                except (ValueError, TypeError): penalty_val = 1
                final_common_mistakes.append({"id": mistake["id"], "description": mistake.get("description", "Нет описания ошибки"), "penalty": penalty_val}); ids_added.add(mistake["id"])
            elif isinstance(mistake, str) and mistake not in [m["description"] for m in final_common_mistakes]:
                 final_common_mistakes.append({"id": f"custom_text_mistake_{len(final_common_mistakes)}", "description": mistake, "penalty": 1})
        generated_scenario["common_mistakes"] = final_common_mistakes

        if age_range_str and age_range_str != "Любой" and 'patient_initial_info_display' in generated_scenario:
            min_a, max_a = AGE_RANGES[age_range_str]
            age_match = re.search(r'\b(\d+)\s*(год|года|лет)?\b', generated_scenario['patient_initial_info_display'], re.IGNORECASE)
            if age_match and not (min_a <= int(age_match.group(1)) <= max_a): st.warning(f"Возраст пациента ({age_match.group(1)}), сгенерированный LLM, не соответствует запрошенному диапазону ({age_range_str}).")
            elif not age_match: st.warning(f"Не удалось извлечь возраст из информации о пациенте ('{generated_scenario['patient_initial_info_display']}'). Запрошенный диапазон: {age_range_str}.")
        if gender_str and gender_str != "Любой" and 'patient_initial_info_display' in generated_scenario:
            info_l = generated_scenario['patient_initial_info_display'].lower()
            gender_terms_male = ["мужск", "мужчин", "мальчик", "пациент "]
            gender_terms_female = ["женск", "женщин", "девочка", "пациентка"]
            found_gender = (gender_str == "Мужской" and any(term in info_l for term in gender_terms_male)) or \
                           (gender_str == "Женский" and any(term in info_l for term in gender_terms_female))
            if not found_gender and not (gender_str.lower()[:3] in info_l):
                 st.warning(f"Запрошенный пол ('{gender_str}') может не соответствовать информации о пациенте, сгенерированной LLM ('{generated_scenario['patient_initial_info_display']}').")
        st.success("Сценарий успешно сгенерирован LLM!"); return generated_scenario
    except (json.JSONDecodeError, ValueError) as e: st.error(f"Ошибка парсинга JSON от LLM: {e}"); st.text_area("Необработанный ответ LLM:", raw_text, height=200); return None
    except Exception as e: st.error(f"Произошла общая ошибка при генерации сценария: {e}"); st.text_area("Необработанный ответ LLM:", raw_text, height=150); return None

def evaluate_with_llm(scenario_data, user_dialogue_msgs, user_dx, user_plan, user_diff_dx="", time_taken_seconds=None, timer_was_active=False, consultations_count=0):
    st.info("Отправка данных LLM-оценщику для анализа..."); raw_text = ""
    scenario_info = f"Сценарий: {scenario_data.get('name', 'N/A')} (Сложность: {scenario_data.get('difficulty_level_tag', 'N/A')})\n" \
                    f"Первичная информация о пациенте: {scenario_data.get('patient_initial_info_display', 'N/A')}\n" \
                    f"Истинный диагноз (детально): {scenario_data.get('true_diagnosis_detailed', 'N/A')}\n" \
                    f"Ключевые моменты анамнеза (эталон): {'; '.join(scenario_data.get('key_anamnesis_points', ['N/A']))}\n" \
                    f"Ожидаемые дифференциальные диагнозы (эталон): {', '.join(scenario_data.get('expected_differential_diagnoses', ['N/A']))}\n" \
                    f"Правильный план (эталон): {scenario_data.get('correct_plan_detailed', 'N/A')}\n" \
                    f"Ключевые слова для проверки диагноза (эталон): {', '.join(scenario_data.get('correct_diagnosis_keywords_for_check', ['N/A']))}\n" \
                    f"Ключевые слова для проверки плана (эталон): {', '.join(scenario_data.get('correct_plan_keywords_for_check', ['N/A']))}\n" \
                    f"Типичные ошибки для данного сценария: {'; '.join([m.get('description', 'N/A') for m in scenario_data.get('common_mistakes', []) if isinstance(m, dict)])}\n"

    dialogue_history_str = ""
    for i, m in enumerate(user_dialogue_msgs):
        role_translated = "Врач" if m['role'] == 'user' else "Пациент"
        dialogue_history_str += f"Ход {i+1} {role_translated}: {m['content']}\n"

    physician_summary = f"Предложенный врачом дифференциальный диагноз: {user_diff_dx or '[Не указан]'}\n" \
                        f"Предложенный врачом окончательный диагноз: {user_dx or '[Отсутствует]'}\n" \
                        f"Предложенный врачом план обследования и лечения: {user_plan or '[Отсутствует]'}\n" \
                        f"Использовано консультаций со специалистом: {consultations_count}\n"
    if timer_was_active and time_taken_seconds is not None: physician_summary += f"Затраченное время: {int(time_taken_seconds // 60)} мин {int(time_taken_seconds % 60)} сек\n"

    system_prompt = "Ты — строгий, но справедливый медицинский преподаватель-эксперт. Твоя задача — оценить работу врача по предоставленному клиническому случаю и его действиям. Предоставь свой ответ СТРОГО в формате JSON. Оценивай объективно, сравнивая действия врача с эталонными данными сценария. Учитывай выбранный уровень сложности сценария."
    user_prompt = f"ДАННЫЕ ЭТАЛОННОГО СЦЕНАРИЯ:\n{scenario_info}\n\nДЕЙСТВИЯ ВРАЧА (ДИАЛОГ И РЕШЕНИЯ):\n{dialogue_history_str}{physician_summary}\n\nЗАДАЧА: Предоставь детальную оценку работы врача в формате JSON. JSON должен иметь следующую структуру:\n" \
                  f"{{\"overall_score\": \"int (общая оценка от 0 до 10)\", \"score_breakdown\": {{\"anamnesis_collection\": {{\"score\": \"int (0-10)\", \"comments\": \"str (комментарии по сбору анамнеза)\"}}, " \
                  f"\"physical_examination\": {{\"score\": \"int (0-10)\", \"comments\": \"str (комментарии по физикальному осмотру)\"}}, \"diagnostic_reasoning\": {{\"score\": \"int (0-10)\", \"comments\": \"str (комментарии по диагностическому мышлению, включая диф.диагноз)\"}}, " \
                  f"\"final_diagnosis_accuracy\": {{\"score\": \"int (0-10)\", \"comments\": \"str (комментарии по точности окончательного диагноза)\"}}, \"treatment_and_management_plan\": {{\"score\": \"int (0-10)\", \"comments\": \"str (комментарии по плану обследования и лечения)\"}}, " \
                  f"\"communication_skills\": {{\"score\": \"int (0-10)|null (если не оценивалось отдельно)\", \"comments\": \"str (комментарии по коммуникативным навыкам)\"}}}}, " \
                  f"\"identified_scenario_mistakes_ids\": [\"list_of_str (ID типичных ошибок из сценария, если были допущены)\"], \"general_feedback\": {{\"positive_aspects\": [\"list_of_str (что было сделано хорошо)\"], \"areas_for_improvement\": [\"list_of_str (что можно улучшить)\"]}}, " \
                  f"\"time_management_comment\": \"str (комментарий по управлению временем, если применимо)\", \"consultation_impact_comment\": \"str (комментарий по влиянию консультаций на оценку, если были использованы. Если консультаций не было, укажи 'Консультации не использовались.')\"}}" \
                  f"\n\nВАЖНО ПО КОНСУЛЬТАЦИЯМ: Если врач использовал консультации ({consultations_count} раз(а)), учти это. Каждая консультация, особенно если случай не был чрезмерно сложным, может свидетельствовать о недостаточной самостоятельности и должна немного снизить оценку за диагностическое мышление (diagnostic_reasoning) и/или общую оценку (overall_score), например, на 0.5-1 балл за каждую использованную консультацию из 10. Отрази это в 'consultation_impact_comment'."

    default_error_result = {"overall_score": 0, "score_breakdown": {}, "general_feedback": {"positive_aspects": [], "areas_for_improvement": ["Произошла ошибка при обработке ответа от LLM-оценщика."]}, "time_management_comment": "Ошибка оценки времени.", "consultation_impact_comment": "Ошибка оценки влияния консультаций."}
    try:
        response = client.chat.completions.create(
            model="local-model", 
            messages=[{"role":"system", "content":system_prompt}, {"role":"user", "content":user_prompt}], 
            max_tokens=4000, 
            temperature=0.3,
            timeout=LLM_TIMEOUT_LONG
        )
        if not (response.choices and response.choices[0].message and response.choices[0].message.content):
            st.error("LLM-оценщик не вернул контент."); return default_error_result
        raw_text = response.choices[0].message.content.strip()
        eval_obj = _extract_and_parse_json(raw_text)

        final_eval = {}
        try: final_eval["overall_score"] = max(0, min(10, int(float(str(eval_obj.get("overall_score",0)).replace(",",".")))))
        except (ValueError, TypeError): final_eval["overall_score"] = 0; st.warning("Некорректное значение 'overall_score' от LLM-оценщика.")

        final_eval["score_breakdown"] = {}
        for cat in ["anamnesis_collection", "physical_examination", "diagnostic_reasoning", "final_diagnosis_accuracy", "treatment_and_management_plan", "communication_skills"]:
            cat_data = eval_obj.get("score_breakdown", {}).get(cat, {})
            score = cat_data.get("score"); val = 0
            if score is not None:
                try: val = max(0, min(10, int(float(str(score).replace(",",".")))))
                except (ValueError, TypeError): val = 0; st.warning(f"Некорректное значение оценки для категории '{cat}'.")
            elif cat == "communication_skills": val = None
            final_eval["score_breakdown"][cat] = {"score": val, "comments": str(cat_data.get("comments", "Комментарии отсутствуют."))}

        mist_ids_raw = eval_obj.get("identified_scenario_mistakes_ids", [])
        final_eval["identified_scenario_mistakes_ids"] = [str(m) for m in mist_ids_raw if isinstance(m, (str,int))] if isinstance(mist_ids_raw, list) else []

        fb_raw = eval_obj.get("general_feedback", {})
        final_eval["general_feedback"] = {
            "positive_aspects": [str(p) for p in fb_raw.get("positive_aspects",[]) if isinstance(p,(str,int,float))] if isinstance(fb_raw.get("positive_aspects"),list) else [],
            "areas_for_improvement": [str(a) for a in fb_raw.get("areas_for_improvement",[]) if isinstance(a,(str,int,float))] if isinstance(fb_raw.get("areas_for_improvement"),list) else []}
        final_eval["time_management_comment"] = str(eval_obj.get("time_management_comment", "Комментарий по тайм-менеджменту отсутствует."))
        final_eval["consultation_impact_comment"] = str(eval_obj.get("consultation_impact_comment", "Комментарий по использованию консультаций отсутствует."))
        st.success("Оценка успешно получена от LLM!"); return final_eval
    except (json.JSONDecodeError, ValueError) as e: st.error(f"Ошибка парсинга JSON от LLM-оценщика: {e}"); st.text_area("Необработанный ответ LLM-оценщика:", raw_text, height=200); return default_error_result
    except Exception as e: st.error(f"Произошла общая ошибка при оценке LLM: {e}"); st.text_area("Необработанный ответ LLM-оценщика:", raw_text, height=150); return default_error_result

default_session_state_values = {
    "scenario_selected": False, "training_mode_active": False, "already_offered_training_mode_for_this_eval": False,
    "start_with_hints_checkbox": False, "evaluation_done": False, "evaluation_results": None, "messages": [], "chat_active": True,
    "current_scenario": None, "user_diagnosis": "", "user_action_plan": "", "user_differential_diagnosis": "",
    "llm_age": "Любой", "llm_gender": "Любой", "llm_spec": "Любая", "llm_difficulty": "Средний",
    "timer_enabled_by_user": False, "timer_duration_setting": 20, "timer_active_in_scenario": False,
    "timer_start_time": None, "time_remaining": None, "timer_expired_flag": False,
    "physician_notes": "", "pending_investigation_results": {}, "current_turn_number": 0,
    "session_history": [], "patient_state_modifiers": [], "app_initialized": False,
    "consultations_used_count": 0, "all_consultation_history": []
}

if not st.session_state.get("app_initialized", False):
    for key, value in default_session_state_values.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.session_state.app_initialized = True

def initialize_scenario(scenario_data_obj, training_mode=False, keep_results_for_training=False):
    if not isinstance(scenario_data_obj, dict):
        st.error("Ошибка: получен неверный тип данных для сценария. Пожалуйста, перезапустите."); st.rerun(); return
    st.session_state.current_scenario = scenario_data_obj
    st.session_state.messages = [{"role": "assistant", "content": scenario_data_obj.get("initial_patient_greeting", "Здравствуйте, доктор.")}]
    if not (training_mode and keep_results_for_training and st.session_state.get("evaluation_results")):
        st.session_state.user_diagnosis = ""
        st.session_state.user_action_plan = ""
        st.session_state.user_differential_diagnosis = ""
        st.session_state.physician_notes = ""
    st.session_state.update({
        "chat_active": True,
        "scenario_selected": True,
        "current_turn_number": 0,
        "pending_investigation_results": {},
        "patient_state_modifiers": [],
        "consultations_used_count": 0,
        "all_consultation_history": []
    })
    if st.session_state.get("timer_enabled_by_user", False):
        st.session_state.update({
            "timer_active_in_scenario": True,
            "timer_start_time": time.time(),
            "time_remaining": st.session_state.get("timer_duration_setting", 20) * 60,
            "timer_expired_flag": False
        })
    else:
        st.session_state.update({
            "timer_active_in_scenario": False,
            "timer_start_time": None,
            "time_remaining": None,
            "timer_expired_flag": False
        })
    st.session_state.training_mode_active = training_mode
    if not training_mode or not keep_results_for_training:
        st.session_state.evaluation_done = False
        st.session_state.evaluation_results = None

def reset_session_and_rerun():
    settings_keys = [
        "llm_age", "llm_gender", "llm_spec", "llm_difficulty",
        "start_with_hints_checkbox",
        "timer_enabled_by_user", "timer_duration_setting"
    ]
    saved_settings = {k: st.session_state.get(k, default_session_state_values.get(k)) for k in settings_keys} 
    history = st.session_state.get("session_history", [])

    for key, value in default_session_state_values.items():
        st.session_state[key] = value

    for k_saved, v_saved in saved_settings.items():
        st.session_state[k_saved] = v_saved

    st.session_state.session_history = history
    st.session_state.app_initialized = True 
    st.rerun()

st.set_page_config(layout="wide", page_title="Виртуальный пациент v2.0");

with st.sidebar:
    st.title("👨‍⚕️ Управление")
    if st.session_state.get("current_scenario"):
        st.success(f"Активен: {st.session_state.current_scenario.get('name', 'Сценарий')[:30]}...")
        if st.session_state.get("timer_active_in_scenario"):
            time_str = get_time_remaining_str()
            if st.session_state.get("timer_expired_flag"): st.error("⏱️ Время вышло!")
            elif time_str != "N/A": st.info(f"⏱️ Осталось: {time_str}")
        if st.button("🔄 Новый сценарий / Сброс", use_container_width=True, type="primary"): reset_session_and_rerun()
        st.markdown("---")

    st.header("⚙️ Параметры")
    with st.expander("Параметры генерации сценария (LLM)", expanded=not st.session_state.get("scenario_selected")):

        selectbox_configs = {
            "llm_age": ("Возраст:", ["Любой"] + list(AGE_RANGES.keys()), 0),
            "llm_gender": ("Пол:", GENDERS, 0),
            "llm_spec": ("Специализация:", ["Любая"] + MEDICAL_SPECIALIZATIONS, 0),
            "llm_difficulty": ("Сложность:", DIFFICULTY_LEVELS, 1)
        }
        for key, (label, options, default_idx_config) in selectbox_configs.items():
            current_value = st.session_state.get(key)
            idx_to_use = options.index(current_value) if current_value in options else default_idx_config
            st.selectbox(label, options, key=key, index=idx_to_use)

        st.checkbox(
            "Режим обучения (с подсказками)",
            key="start_with_hints_checkbox",
            value=st.session_state.get("start_with_hints_checkbox", default_session_state_values["start_with_hints_checkbox"])
        )

    with st.expander("⏱️ Настройки таймера"):
        st.checkbox(
            "Включить таймер для сценария",
            key="timer_enabled_by_user",
            value=st.session_state.get("timer_enabled_by_user", default_session_state_values["timer_enabled_by_user"])
        )
        if st.session_state.timer_enabled_by_user:
            current_duration_val = st.session_state.get("timer_duration_setting", default_session_state_values["timer_duration_setting"])
            default_timer_idx = TIMER_DURATIONS_MINUTES.index(current_duration_val) if current_duration_val in TIMER_DURATIONS_MINUTES else 2
            st.selectbox(
                "Длительность (минуты):",
                TIMER_DURATIONS_MINUTES,
                key="timer_duration_setting",
                index=default_timer_idx
            )

    if st.button("✨ Сгенерировать новый сценарий (LLM)", use_container_width=True, type="primary"):
        if st.session_state.get("scenario_selected"):
             reset_session_and_rerun() 
        
        with st.spinner("LLM генерирует новый сценарий... Это может занять до 5 минут. Пожалуйста, подождите."):
            new_scenario = generate_new_scenario_via_llm(
                st.session_state.llm_age,
                st.session_state.llm_spec,
                st.session_state.llm_gender,
                st.session_state.llm_difficulty
            )
        if new_scenario:
            initialize_scenario(new_scenario, st.session_state.start_with_hints_checkbox)
            st.session_state.already_offered_training_mode_for_this_eval = False
            st.rerun()
        else:
            st.error("Не удалось сгенерировать сценарий. Попробуйте еще раз или проверьте настройки и доступность LLM сервера. Возможно, истек таймаут запроса к LLM.")

    st.markdown("---"); st.caption(f"LLM API: {KOBOLD_API_URL.replace('http://localhost', 'local')[:50]}...")


if st.session_state.get("current_scenario") and st.session_state.get("scenario_selected"):
    scenario = st.session_state.current_scenario; is_training = st.session_state.training_mode_active
    if not isinstance(scenario, dict): reset_session_and_rerun(); st.stop()

    if st.session_state.timer_active_in_scenario and \
       not st.session_state.evaluation_done and \
       not st.session_state.timer_expired_flag:
        if st.session_state.timer_start_time:
            elapsed_time = time.time() - st.session_state.timer_start_time
            st.session_state.time_remaining = max(0, (st.session_state.timer_duration_setting * 60) - elapsed_time)
            if st.session_state.time_remaining <= 0:
                st.session_state.timer_expired_flag = True
                st.session_state.chat_active = False
                st.error("Время сеанса истекло! Пожалуйста, завершите сценарий и получите оценку.")
        time.sleep(0.5); st.rerun()

    st.title(f"🩺 {scenario.get('name', 'Клинический случай без названия')}")
    timer_disp_html = "";
    if st.session_state.timer_active_in_scenario or st.session_state.timer_expired_flag or (st.session_state.evaluation_done and st.session_state.get('time_taken_for_display') is not None):
        time_str_display = get_time_remaining_str()
        if st.session_state.timer_expired_flag:
            timer_disp_html = "<p style='margin-bottom:0 !important;'><strong>⏱️ Время:</strong> <span style='color:red;'>Истекло!</span></p>"
        elif time_str_display != "N/A" and st.session_state.timer_active_in_scenario :
             timer_disp_html = f"<p style='margin-bottom:0 !important;'><strong>⏱️ Осталось:</strong> {time_str_display}</p>"
        elif st.session_state.evaluation_done and st.session_state.get('time_taken_for_display'):
            timer_disp_html = f"<p style='margin-bottom:0 !important;'><strong>⏱️ Затрачено:</strong> {st.session_state.get('time_taken_for_display')}</p>"

    mode_status = ' (Завершено)' if st.session_state.evaluation_done else ' (В процессе)'
    mode_text = f"{'💡' if is_training else '🚀'} {'Режим обучения' if is_training else 'Самостоятельная работа'}{mode_status}"

    st.markdown(f"""<div class="info-card" style="background-color:#F0F2F6;border:1px solid #E0E0E0;padding:12px 18px;border-radius:8px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
        <p style="margin-bottom:6px !important;font-size:0.95rem;"><strong>Пациент:</strong> {scenario.get('patient_initial_info_display', 'Информация отсутствует.')}</p>
        <p style="margin-bottom:6px !important;font-size:0.95rem;"><strong>Сложность:</strong> {scenario.get('difficulty_level_tag', 'Не указана')}</p>
        <p style="margin-bottom:6px !important;font-size:0.95rem;"><strong>Режим:</strong> {mode_text}</p>{timer_disp_html}</div>""", unsafe_allow_html=True)

    tab_titles = ["Диалог и Действия", "Записная книжка"]
    tab_icons = {"Диалог и Действия": "💬", "Записная книжка": "📝"}
    if st.session_state.evaluation_done: tab_titles.append("Результаты Оценки"); tab_icons["Результаты Оценки"] = "📊"
    if st.session_state.session_history and st.session_state.evaluation_done : tab_titles.append("История сессий"); tab_icons["История сессий"] = "📚" 
    if is_training or st.session_state.evaluation_done: tab_titles.append("Детали Сценария (Подсказки)"); tab_icons["Детали Сценария (Подсказки)"] = "ℹ️"

    active_tabs_map = {}
    current_tab_idx = 0
    for title in ["Диалог и Действия", "Записная книжка"]:
        active_tabs_map[title] = current_tab_idx
        current_tab_idx +=1

    if "Результаты Оценки" in tab_titles:
        active_tabs_map["Результаты Оценки"] = current_tab_idx
        current_tab_idx +=1
    if "История сессий" in tab_titles:
        active_tabs_map["История сессий"] = current_tab_idx
        current_tab_idx +=1
    if "Детали Сценария (Подсказки)" in tab_titles:
        active_tabs_map["Детали Сценария (Подсказки)"] = current_tab_idx
        current_tab_idx+=1

    tabs_rendered = st.tabs([f"{tab_icons.get(t, '')} {t}" for t in tab_titles])

    with tabs_rendered[active_tabs_map["Диалог и Действия"]]:
        col1_main, col2_main = st.columns([0.45, 0.55])
        with col1_main:
            st.subheader("👨‍⚕️ Ваши действия и решения")
            inputs_disabled_status = st.session_state.evaluation_done or \
                                     (st.session_state.timer_active_in_scenario and \
                                      st.session_state.timer_expired_flag and \
                                      not st.session_state.evaluation_done)

            with st.form(key="dx_plan_submission_form"):
                st.text_area("Ваш предполагаемый дифференциальный диагноз:", value=st.session_state.user_differential_diagnosis, key="user_diff_dx_input_main", disabled=inputs_disabled_status, height=100, help="Перечислите возможные диагнозы через запятую.")
                st.text_input("Ваш окончательный диагноз:", value=st.session_state.user_diagnosis, key="user_dx_input_main", disabled=inputs_disabled_status, help="Укажите наиболее вероятный диагноз.")
                st.text_area("Ваш план обследования и лечения:", value=st.session_state.user_action_plan, key="user_plan_input_main", height=150, disabled=inputs_disabled_status, help="Опишите дальнейшие шаги.")

                submit_button_label = "✔️ Завершить сценарий и получить оценку"
                if st.form_submit_button(submit_button_label, type="primary", use_container_width=True, disabled=st.session_state.evaluation_done):
                    st.session_state.update({
                        "user_differential_diagnosis": st.session_state.user_diff_dx_input_main,
                        "user_diagnosis": st.session_state.user_dx_input_main,
                        "user_action_plan": st.session_state.user_plan_input_main,
                        "chat_active": False
                    })

                    time_taken_final = None
                    if st.session_state.timer_active_in_scenario or st.session_state.timer_expired_flag :
                        if st.session_state.timer_expired_flag:
                            time_taken_final = st.session_state.timer_duration_setting * 60
                        elif st.session_state.timer_start_time:
                             time_taken_final = time.time() - st.session_state.timer_start_time

                    if time_taken_final is not None:
                         st.session_state.time_taken_for_display = f"{int(time_taken_final//60)}:{int(time_taken_final%60):02d}"

                    with st.spinner("LLM проводит оценку ваших действий... Это может занять до 5 минут."):
                        eval_results = evaluate_with_llm(scenario, st.session_state.messages,
                                                    st.session_state.user_diagnosis, st.session_state.user_action_plan,
                                                    st.session_state.user_differential_diagnosis,
                                                    time_taken_final,
                                                    (st.session_state.timer_active_in_scenario or st.session_state.timer_expired_flag),
                                                    st.session_state.get("consultations_used_count", 0)
                                                    )
                    st.session_state.evaluation_results = eval_results
                    if eval_results and "overall_score" in eval_results:
                        st.session_state.session_history.append({
                            "name": scenario.get("name", "N/A"),
                            "score": eval_results["overall_score"],
                            "difficulty": scenario.get("difficulty_level_tag", "N/A"),
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "time_taken": st.session_state.time_taken_for_display if time_taken_final is not None else "N/A",
                            "timer_active": (st.session_state.timer_active_in_scenario or st.session_state.timer_expired_flag),
                            "consultations_used": st.session_state.get("consultations_used_count", 0)
                        })
                        if len(st.session_state.session_history) > 20:
                            st.session_state.session_history = st.session_state.session_history[-20:]

                    st.session_state.evaluation_done = True
                    st.session_state.timer_active_in_scenario = False 
                    st.rerun()

            st.markdown("---")
            st.subheader("🤝 Консультация со специалистом")

            consult_section_disabled = st.session_state.evaluation_done or \
                               (st.session_state.timer_active_in_scenario and st.session_state.timer_expired_flag and not st.session_state.evaluation_done)

            consult_limit_reached = st.session_state.consultations_used_count >= MAX_CONSULTATIONS

            if st.session_state.all_consultation_history:
                with st.expander("История консультаций по этому случаю", expanded=False):
                    for i, consult_item in enumerate(reversed(st.session_state.all_consultation_history)): 
                        st.markdown(f"**Консультация {len(st.session_state.all_consultation_history) - i} (Специалист: {consult_item['specialist']})**")
                        st.markdown(f"> *Ваш вопрос:* `{consult_item['request']}`")
                        st.markdown(f"> *Ответ консультанта:* {consult_item['response']}")
                        st.divider()

            if consult_limit_reached and not consult_section_disabled :
                st.warning(f"Вы использовали все доступные консультации ({MAX_CONSULTATIONS}).")

            if not consult_section_disabled and not consult_limit_reached:
                with st.form(key="consultation_form"):
                    st.info(f"Доступно консультаций: {MAX_CONSULTATIONS - st.session_state.consultations_used_count} из {MAX_CONSULTATIONS}. Каждая консультация может повлиять на итоговую оценку.")

                    consult_specialist = st.selectbox(
                        "Выберите специализацию консультанта:",
                        MEDICAL_SPECIALIZATIONS,
                        key="consult_specialist_choice"
                    )
                    consult_question = st.text_area(
                        "Ваш вопрос консультанту (сформулируйте четко и кратко):",
                        key="consult_question_text",
                        height=100,
                        placeholder="Например: 'Пациент N, диагноз X. Неясна тактика по Y. Ваше мнение?'"
                    )
                    submit_consult_button = st.form_submit_button(
                        "📨 Отправить запрос на консультацию",
                        disabled=not consult_question.strip() 
                    )

                    if submit_consult_button and consult_question.strip():
                        st.session_state.consultations_used_count += 1

                        with st.spinner("Получение ответа от консультанта... Это может занять до 1 минуты."):
                            main_scenario_info_for_consultant = scenario.get('patient_initial_info_display', 'Информация отсутствует.')
                            consultant_advice = get_consultant_response(
                                st.session_state.messages,
                                consult_question,
                                consult_specialist,
                                main_scenario_info_for_consultant
                            )

                        st.session_state.all_consultation_history.append({
                            "specialist": consult_specialist,
                            "request": consult_question,
                            "response": consultant_advice
                        })
                        st.success("Ответ от консультанта получен и добавлен в историю консультаций.")
                        st.rerun()
            elif not consult_section_disabled and consult_limit_reached:
                 pass
            elif consult_section_disabled :
                st.info("Консультации недоступны на данном этапе (сценарий завершен / время истекло).")


        with col2_main:
            st.subheader("💬 Диалог с пациентом")
            chat_interface_disabled = not st.session_state.chat_active or \
                                      st.session_state.evaluation_done or \
                                      (st.session_state.timer_active_in_scenario and st.session_state.timer_expired_flag)

            if not chat_interface_disabled:
                st.markdown("<small>Быстрые действия (физикальный осмотр):</small>", unsafe_allow_html=True);
                quick_actions_map = {
                    "t°": "Измеряю температуру.", "АД": "Измеряю АД.", "ЧСС": "Измеряю ЧСС.", "SpO2": "Проверяю SpO2.",
                    "Лёгкие": "Слушаю лёгкие.", "Живот": "Пальпирую живот.", "Горло": "Осматриваю горло.",
                    "Кожа": "Осмотр кожных покровов.", "Л/У": "Пальпация лимфоузлов.", "Щитовидка": "Пальпация щитовидной железы.",
                    "Суставы": "Осмотр и пальпация суставов.", "Невро-С (кр)": "Краткий неврологический осмотр.",
                    "Печень/Сел": "Пальпация печени и селезенки.", "Отеки": "Проверка на отеки.", "Пульс (периф)": "Проверка периферической пульсации.",
                    "Уши": "Осмотр ушей.", "Нос": "Осмотр носа."
                }
                action_to_phys_key_approx = {
                    "t°": ["temperature", "температура"], "ад": ["blood_pressure", "давление", "ад"],
                    "чсс": ["heart_rate", "pulse", "чсс"], "spo2": ["spo2", "сатурация"],
                    "лёгкие": ["auscultation_lungs", "легкие", "дыхание", "перкуссия легких", "percussion_lungs"],
                    "живот": ["palpation_abdomen", "живот", "абдоминальная пальпация"],
                    "горло": ["throat_inspection", "throat", "горло", "зев", "миндалины", "осмотр рта", "tongue_appearance"],
                    "кожа": ["skin_appearance", "skin_general", "кожные покровы", "осмотр кожи"],
                    "л/у": ["lymph_nodes", "peripheral_lymph_nodes", "лимфоузлы"],
                    "щитовидка": ["thyroid_palpation", "щитовидная железа"],
                    "суставы": ["joints_inspection", "joints_palpation", "суставы"],
                    "невро-с (кр)": ["neuro_status_brief", "neurological_status_brief", "неврологический статус кратко", "сознание", "зрачки", "речь"],
                    "печень/сел": ["liver_palpation", "spleen_palpation", "печень", "селезенка"],
                    "отеки": ["edema_check", "peripheral_edema", "отеки"],
                    "пульс (периф)": ["peripheral_pulses", "периферическая пульсация"],
                    "уши": ["ear_inspection", "otoscopy", "уши", "осмотр ушей"],
                    "нос": ["nose_inspection", "rhinoscopy", "нос", "осмотр носа"]
                }

                scenario_phys_exam_keys = [k.lower().strip() for k in scenario.get("physical_exam_findings_prompt_details", {}).keys()]
                relevant_quick_actions = []

                if not scenario.get("physical_exam_findings_prompt_details"):
                    relevant_quick_actions = list(quick_actions_map.items())
                else:
                    for label, text_action in quick_actions_map.items():
                        label_key_for_approx = label.lower().replace('°','deg').replace('ё','е').replace('/','_').replace(' (кр)','').replace(' (периф)','')
                        is_relevant = False
                        potential_direct_keys = action_to_phys_key_approx.get(label_key_for_approx, [label_key_for_approx])
                        if any(pk.lower().strip() in scenario_phys_exam_keys for pk in potential_direct_keys):
                            is_relevant = True
                        if is_relevant:
                             relevant_quick_actions.append((label, text_action))

                if not relevant_quick_actions : 
                     relevant_quick_actions = list(quick_actions_map.items())


                action_buttons_cols = st.columns(4) 
                btn_idx = 0
                for Rlabel, Rtext_action in relevant_quick_actions:
                    col_to_use = action_buttons_cols[btn_idx % len(action_buttons_cols)]
                    if col_to_use.button(Rlabel, key=f"quick_action_{Rlabel.replace('°','deg').replace('/','_').replace(' ','_').replace('(','').replace(')','')}", use_container_width=True, help=Rtext_action):
                        st.session_state.messages.append({"role": "user", "content": Rtext_action}); st.session_state.current_turn_number += 1
                        st.session_state.user_input_trigger_flag = True; st.rerun()
                    btn_idx += 1
                
                st.markdown("<small>Быстрые инструментальные назначения:</small>", unsafe_allow_html=True)
                quick_investigations_map = {
                    "ЭКГ": "Назначаю ЭКГ.",
                    "Рентген ОГК": "Назначаю рентген органов грудной клетки.",
                    "УЗИ ОБП": "Назначаю УЗИ органов брюшной полости.",
                    "ОАМ": "Назначаю общий анализ мочи.",
                    "Глюкоза (экспр)": "Проверить глюкозу крови экспресс-методом."
                }
                investigation_buttons_cols = st.columns(3)
                inv_btn_idx = 0
                for Ilabel, Itext_action in quick_investigations_map.items():
                    col_to_use_inv = investigation_buttons_cols[inv_btn_idx % len(investigation_buttons_cols)]
                    if col_to_use_inv.button(Ilabel, key=f"quick_investigation_{Ilabel.replace(' ','_').replace('(','').replace(')','')}", use_container_width=True, help=Itext_action):
                        st.session_state.messages.append({"role": "user", "content": Itext_action})
                        st.session_state.current_turn_number += 1
                        st.session_state.user_input_trigger_flag = True 
                        st.rerun()
                    inv_btn_idx +=1
                st.markdown("---")

            chat_container_height = 360 if not chat_interface_disabled else 520 
            with st.container(height=chat_container_height):
                for msg_idx, msg_item in enumerate(st.session_state.messages):
                    avatar_icon = "🧑‍⚕️" if msg_item["role"] == "user" else "🤒"
                    with st.chat_message(msg_item["role"], avatar=avatar_icon): st.markdown(msg_item["content"])

            user_typed_query = st.chat_input("Задайте вопрос пациенту или опишите действие...", key="main_chat_input_field", disabled=chat_interface_disabled)

            if user_typed_query or st.session_state.pop("user_input_trigger_flag", False):
                if user_typed_query:
                    st.session_state.messages.append({"role": "user", "content": user_typed_query})
                    st.session_state.current_turn_number += 1

                last_user_message_content = st.session_state.messages[-1]["content"].lower() if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" else ""

                if "available_investigations" in scenario and last_user_message_content:
                    for inv_key, inv_details in scenario["available_investigations"].items():
                        if isinstance(inv_details,dict) and any(keyword.lower() in last_user_message_content for keyword in inv_details.get("request_keywords",[])):
                            if inv_key not in st.session_state.pending_investigation_results or not st.session_state.pending_investigation_results[inv_key]["provided"]:
                                st.session_state.pending_investigation_results[inv_key] = {
                                    "ready_at_turn": st.session_state.current_turn_number + inv_details.get("turn_to_provide_results",1),
                                    "results_text": inv_details.get("results_text", "Результаты данного исследования обрабатываются..."),
                                    "provided": False
                                }
                                st.toast(f"Исследование '{inv_key}' было назначено.", icon="⏳")

                base_persona_prompt = scenario.get("patient_llm_persona_system_prompt", "Ты пациент. Отвечай на вопросы врача.")
                difficulty_tag = scenario.get("difficulty_level_tag", "Средний")
                difficulty_modifier = DIFFICULTY_LEVELS_DETAILS.get(difficulty_tag, {}).get("patient_persona_modifier", "")

                phys_exam_details_str = json.dumps(scenario.get("physical_exam_findings_prompt_details", {}), ensure_ascii=False)
                avail_inv_details_str = json.dumps(scenario.get("available_investigations", {}), ensure_ascii=False)

                general_simulation_instructions = f"""
ОБЩИЕ ИНСТРУКЦИИ ПО СИМУЛЯЦИИ:
- Твоя главная задача: отвечать на вопросы врача, предоставлять информацию о своем самочувствии, истории болезни, а также симулировать результаты осмотров и анализов, когда врач их запрашивает.
- ФИЗИКАЛЬНЫЙ ОСМОТР: Если врач пишет что-то вроде 'измеряю АД', 'слушаю легкие', 'пальпирую живот в такой-то области', 'осматриваю горло', 'проверяю кожные покровы', 'пальпирую лимфоузлы' и т.п., ты должен СИМУЛИРОВАТЬ результат этого действия и ВКЛЮЧИТЬ ЕГО В СВОЙ ОТВЕТ. Результаты должны быть клинически правдоподобными и соответствовать твоему состоянию в сценарии. Используй следующие данные для симуляции осмотра (если врач запросит что-то из этого, предоставь соответствующее значение): {phys_exam_details_str}. Если врач просит осмотреть что-то, чего нет в этих деталях, но это логично для твоего состояния, дай правдоподобный ответ. Если врач просто пишет 'Проведу осмотр', уточни: 'Что именно Вы хотите осмотреть, доктор?' Не предлагай провести осмотр сам.
- АНАЛИЗЫ И ИССЛЕДОВАНИЯ: Если врач ЗАПРАШИВАЕТ какое-либо исследование, название которого (или его ключевые слова) совпадает с одним из доступных исследований (см. ниже), ты должен подтвердить назначение, но НЕ сообщать результаты сразу. Результаты станут "готовы" через указанное количество ходов ('turn_to_provide_results') и будут автоматически добавлены к твоему ответу приложением. Просто подтверди, что исследование будет сделано, например: 'Хорошо, доктор, я сдам этот анализ'. Доступные исследования для этого сценария: {avail_inv_details_str}.
- ПОМНИ: Ты не должен сам предлагать провести осмотр или назначить анализы. Реагируй только на действия врача.
"""
                current_patient_state_modifiers = "\n".join(st.session_state.patient_state_modifiers)
                final_system_prompt_for_patient = f"{base_persona_prompt}\n{difficulty_modifier}\n{general_simulation_instructions}\n{current_patient_state_modifiers}"

                with st.spinner("Пациент обдумывает ответ... Это может занять до 1 минуты."):
                    llm_patient_response = generate_llm_response(st.session_state.messages, final_system_prompt_for_patient)

                response_parts_combined = [llm_patient_response]
                for inv_key, inv_status_data in sorted(st.session_state.pending_investigation_results.items(), key=lambda x_item: x_item[1]['ready_at_turn']):
                    if not inv_status_data["provided"] and st.session_state.current_turn_number >= inv_status_data["ready_at_turn"]:
                        response_parts_combined.append(f"\n\n📋 **Результаты исследования '{inv_key}':**\n{inv_status_data['results_text']}")
                        st.session_state.pending_investigation_results[inv_key]["provided"] = True
                        st.toast(f"Получены результаты исследования '{inv_key}'!", icon="📄")
                st.session_state.messages.append({"role": "assistant", "content": "".join(response_parts_combined)})

                if "dynamic_state_triggers" in scenario:
                    for trigger_item in scenario.get("dynamic_state_triggers",[]):
                        if isinstance(trigger_item, dict) and \
                           trigger_item.get("condition_type") == "missed_key_question" and \
                           trigger_item.get("key_question_keyword") and \
                           trigger_item.get("turns_to_trigger") == st.session_state.current_turn_number and \
                           not any(trigger_item["key_question_keyword"].lower() in m_item["content"].lower() for m_item in st.session_state.messages if m_item["role"]=="user") and \
                           trigger_item.get("patient_response_cue"):
                            st.session_state.messages.append({"role": "assistant", "content": f"(Пациент внезапно вспоминает) {trigger_item['patient_response_cue']}"})
                            st.session_state.patient_state_modifiers.append(f"Пациент дополнительно сообщил следующее: {trigger_item['patient_response_cue']}")
                            st.toast("Пациент что-то вспомнил и добавил информацию!", icon="💡"); break
                st.rerun()
            elif user_typed_query and chat_interface_disabled:
                st.toast("Диалог с пациентом завершен или время вышло. Вы не можете отправить новое сообщение.", icon="ℹ️")

    if "Записная книжка" in tab_titles:
        with tabs_rendered[active_tabs_map["Записная книжка"]]:
            st.subheader("📝 Ваши личные заметки по случаю"); st.caption("Эта информация невидима для пациента и не передается LLM-оценщику.")
            current_notes = st.text_area("Заметки:", value=st.session_state.physician_notes, height=450, key="physician_notes_input_area", help="Здесь вы можете делать любые пометки для себя.")
            if current_notes != st.session_state.physician_notes:
                st.session_state.physician_notes = current_notes; 

    if "Результаты Оценки" in tab_titles:
        with tabs_rendered[active_tabs_map["Результаты Оценки"]]:
            eval_results_data = st.session_state.evaluation_results
            st.subheader(f"📊 Результаты вашей {'учебной сессии' if is_training else 'работы по сценарию'}")
            if eval_results_data and "overall_score" in eval_results_data:
                st.metric(label="🏆 Итоговая оценка (0-10)", value=f"{eval_results_data.get('overall_score',0)}")
                st.markdown(f"**Комментарий по тайм-менеджменту:** {eval_results_data.get('time_management_comment', 'Комментарий отсутствует.')}")
                st.markdown(f"**Комментарий по использованию консультаций:** {eval_results_data.get('consultation_impact_comment', 'Консультации не использовались или комментарий отсутствует.')}")

                st.markdown("---"); st.subheader("🔍 Детализация оценки по компонентам:")
                breakdown_cols, col_idx_b = st.columns(2), 0
                for cat_key, cat_val_data in eval_results_data.get("score_breakdown", {}).items():
                    if not isinstance(cat_val_data, dict): continue
                    cat_name_map = {"anamnesis_collection": "Сбор анамнеза", "physical_examination": "Физикальный осмотр", "diagnostic_reasoning": "Диагностическое мышление",
                                    "final_diagnosis_accuracy": "Точность окончательного диагноза", "treatment_and_management_plan": "План ведения и лечения", "communication_skills": "Коммуникативные навыки"}
                    display_cat_name = cat_name_map.get(cat_key, cat_key.replace("_"," ").capitalize()); score_val, comments_text = cat_val_data.get('score'), cat_val_data.get('comments','Комментарии отсутствуют.')
                    with breakdown_cols[col_idx_b % 2]:
                        score_display_text = '-' if score_val is None else f'{score_val}/10'
                        st.markdown(f"**{display_cat_name}:** {score_display_text}")
                        expand_comment = (score_val is not None and score_val < 9 and score_val !=0) or (comments_text and comments_text.lower() not in ["n/a", "комментарии отсутствуют."])
                        with st.expander(f"Подробнее по '{display_cat_name}'", expanded=expand_comment): st.caption(comments_text)
                    col_idx_b +=1
                st.markdown("---")
                general_fb = eval_results_data.get("general_feedback", {})
                if isinstance(general_fb, dict):
                    with st.expander("👍 Положительные аспекты вашей работы", expanded=True):
                        positive_aspects_list = general_fb.get('positive_aspects',[])
                        if not positive_aspects_list : st.info("Положительные моменты не были выделены LLM-оценщиком.")
                        else:
                            for pa_item in positive_aspects_list: st.success(f"- {pa_item}", icon="👍")
                    with st.expander("🤔 Области для дальнейшего улучшения", expanded=True):
                        areas_for_improvement_list = general_fb.get('areas_for_improvement',[])
                        if not areas_for_improvement_list and eval_results_data.get('overall_score',0) >=9: st.balloons(); st.success("🎉 Отличная работа! Замечаний по улучшению нет.")
                        elif not areas_for_improvement_list: st.info("Области для улучшения не были указаны LLM-оценщиком.")
                        else:
                            for ai_item in areas_for_improvement_list: st.warning(f"- {ai_item}", icon="🤔")

                identified_mistakes_ids = eval_results_data.get("identified_scenario_mistakes_ids", [])
                if identified_mistakes_ids:
                    st.subheader("🚫 Выявленные типичные ошибки из сценария:");
                    scenario_mistakes_map = {m_item["id"]:m_item["description"] for m_item in scenario.get("common_mistakes",[]) if isinstance(m_item,dict)}
                    for m_id in identified_mistakes_ids: st.error(f"- {scenario_mistakes_map.get(m_id, f'Ошибка с ID: {m_id} (описание не найдено в сценарии)')}", icon="🚫")

                st.markdown("---")
                if not is_training and not st.session_state.already_offered_training_mode_for_this_eval:
                    if st.button("💡 Пройти этот сценарий в режиме обучения (с подсказками)", use_container_width=True):
                        st.session_state.already_offered_training_mode_for_this_eval=True
                        initialize_scenario(scenario, True, True); st.rerun()
            else: st.warning("Результаты оценки еще не готовы или произошла ошибка при их получении.")

    if "История сессий" in tab_titles:
         with tabs_rendered[active_tabs_map["История сессий"]]:
            st.subheader("📚 Ваша история пройденных сессий")
            if st.session_state.session_history:
                df_history_display = pd.DataFrame(st.session_state.session_history)
                df_history_display.rename(columns={
                    "name": "Название сценария",
                    "score": "Итоговая оценка",
                    "difficulty": "Уровень сложности",
                    "date": "Дата и время",
                    "time_taken": "Затраченное время",
                    "timer_active": "Таймер был активен",
                    "consultations_used": "Использовано консультаций"
                }, inplace=True)

                expected_cols = ["Название сценария", "Итоговая оценка", "Уровень сложности", "Дата и время", "Затраченное время", "Таймер был активен", "Использовано консультаций"]
                for col_name in expected_cols:
                    if col_name not in df_history_display.columns:
                        df_history_display[col_name] = "N/A"

                df_history_display = df_history_display[expected_cols]
                st.dataframe(df_history_display, hide_index=True, use_container_width=True)

                if st.button("🗑️ Очистить историю сессий", key="clear_session_history_button_active_scenario", help="Это действие удалит всю сохраненную историю сессий."):
                    st.session_state.session_history = []; st.rerun()
            else: st.info("Ваша история сессий пока пуста. Завершите сценарий, чтобы он появился здесь.")

    if "Детали Сценария (Подсказки)" in tab_titles:
        with tabs_rendered[active_tabs_map["Детали Сценария (Подсказки)"]]:
            st.subheader("ℹ️ Эталонные детали текущего сценария")
            st.markdown(f"**Уровень сложности:** {scenario.get('difficulty_level_tag', 'Не указан')}")
            st.markdown(f"**Истинный диагноз (кратко):** `{scenario.get('true_diagnosis_internal', 'Не указан')}`")
            with st.expander("**Истинный диагноз (подробное описание):**", expanded=False): st.info(f"{scenario.get('true_diagnosis_detailed', 'Описание отсутствует.')}")
            st.markdown("**Ключевые моменты анамнеза, которые нужно было выявить:**");
            key_anamnesis_points_list = scenario.get('key_anamnesis_points', [])
            if not key_anamnesis_points_list: st.caption("  - Ключевые моменты анамнеза не указаны в сценарии.")
            else:
                for point_item in key_anamnesis_points_list: st.markdown(f"  - *{str(point_item)}*")

            with st.expander("**Эталонный план обследования и лечения:**", expanded=False): st.success(f"{scenario.get('correct_plan_detailed', 'План не указан.')}")

            with st.expander("**Ожидаемые дифференциальные диагнозы:**", expanded=False):
                expected_diff_dx_list = scenario.get('expected_differential_diagnoses', [])
                if not expected_diff_dx_list: st.caption("  - Ожидаемые дифференциальные диагнозы не указаны.")
                else:
                    for dx_item in expected_diff_dx_list: st.markdown(f"  - {str(dx_item)}")

            with st.expander("**Эталонные данные физикального осмотра (которые пациент мог бы сообщить):**", expanded=False):
                physical_exam_findings_dict = scenario.get('physical_exam_findings_prompt_details', {})
                if not physical_exam_findings_dict: st.caption("  - Данные физикального осмотра не указаны.")
                else:
                    for key_exam, val_exam in physical_exam_findings_dict.items(): st.markdown(f"  - **{key_exam.replace('_',' ').capitalize()}:** {str(val_exam)}")

            with st.expander("**Доступные исследования и их ожидаемые результаты (эталон):**", expanded=False):
                available_investigations_dict = scenario.get('available_investigations', {})
                if not available_investigations_dict: st.caption("  - Доступные исследования не указаны.")
                else:
                    for inv_key_item, inv_data_item in available_investigations_dict.items():
                        if isinstance(inv_data_item, dict) and str(inv_data_item.get("results_text","")).strip():
                            st.markdown(f"  - **{inv_key_item}:** {inv_data_item['results_text']} (Ключевые слова для запроса: `{(', '.join(inv_data_item.get('request_keywords',[])))}`)")

            st.markdown("**Типичные ошибки, возможные в данном сценарии:**")
            common_mistakes_list = scenario.get("common_mistakes",[])
            if not common_mistakes_list: st.caption(" - Типичные ошибки для этого сценария не указаны.")
            else:
                for mistake_dict_item in common_mistakes_list:
                    if isinstance(mistake_dict_item, dict):
                        st.markdown(f"  - {mistake_dict_item.get('description', 'Описание ошибки отсутствует.')} (ID: {mistake_dict_item.get('id','N/A')}, Штрафные баллы: {mistake_dict_item.get('penalty',0)})")


elif not st.session_state.get("current_scenario") and not st.session_state.get("scenario_selected"):
    initial_tab_titles = ["О приложении", "Готовые сценарии", "История сессий"]
    initial_tab_icons = {"О приложении": "👋", "Готовые сценарии": "📚", "История сессий": "📈"}
    
    tab_specs = [f"{initial_tab_icons.get(title, '')} {title}" for title in initial_tab_titles]
    welcome_tab, predef_scenarios_tab, history_tab_initial = st.tabs(tab_specs)


    with welcome_tab:
        st.title("Добро пожаловать в симулятор 'Виртуальный пациент' v2.0!")
        st.markdown("""
        Этот интерактивный тренажер предназначен для отработки навыков сбора анамнеза, проведения виртуального физикального осмотра, назначения и интерпретации "лабораторных и инструментальных исследований", постановки дифференциального и окончательного диагноза, а также формирования плана ведения пациента.
        Вы будете взаимодействовать с пациентом, управляемым большой языковой моделью (LLM).

        **Ключевые возможности:**
        - **Динамический диалог:** Задавайте вопросы пациенту в свободной форме.
        - **Виртуальный осмотр:** "Проводите" физикальный осмотр (используя текстовые команды или быстрые кнопки), и пациент сообщит "результаты".
        - **"Назначение" исследований:** Запрашивайте анализы и исследования (текстом или через быстрые кнопки), получайте "результаты" с задержкой.
        - **Консультации со специалистом:** Возможность "проконсультироваться" с LLM-специалистом (с влиянием на оценку).
        - **Разные уровни сложности:** От легких классических случаев до экспертных с коммуникативными вызовами.
        - **Таймер:** Возможность включить ограничение по времени на сценарий.
        - **Записная книжка:** Делайте личные заметки в ходе сценария.
        - **Детализированная оценка:** Получите подробный разбор ваших действий от LLM-преподавателя.
        - **Режим обучения:** Проходите сценарии с доступом к эталонной информации.
        - **История сессий:** Отслеживайте свой прогресс.
        """)
        st.markdown("---")
        col1_info, col2_info = st.columns(2)
        with col1_info:
            st.subheader("🚀 Как начать работу?")
            st.markdown("""
            1.  **Настройте параметры** в панели слева (возраст, пол, специализация, сложность для LLM-генерации, режим обучения).
            2.  **Выберите настройки таймера** (по желанию).
            3.  **Либо сгенерируйте сценарий с помощью LLM**, нажав кнопку "✨ Сгенерировать новый сценарий (LLM)" в левой панели.
            4.  **Либо перейдите на вкладку "📚 Готовые сценарии"** (см. выше) и выберите один из предложенных.
            5.  После выбора сценария, ознакомьтесь с первичной информацией о пациенте.
            6.  Начните диалог:
                *   Задавайте открытые и закрытые вопросы в текстовом поле.
                *   Используйте кнопки **"Быстрые действия (физикальный осмотр)"** для стандартных элементов осмотра.
                *   Используйте кнопки **"Быстрые инструментальные назначения"** для распространенных исследований.
                *   "Назначайте" другие анализы и исследования, формулируя запрос в чате (например, "Назначаю общий анализ крови" или "Сделаем рентген грудной клетки").
            7.  При необходимости, используйте **"Консультацию со специалистом"** (помните об ограничении и влиянии на оценку).
            8.  Используйте вкладку **"Записная книжка"** для своих пометок.
            9.  Когда будете готовы, введите свой дифференциальный и окончательный диагноз, а также план лечения, и нажмите **"✔️ Завершить сценарий и получить оценку"**.
            """)
        with col2_info:
            st.subheader("🎯 Ваши основные задачи в сценарии:")
            st.markdown("""
            *   **Собрать полный и релевантный анамнез**.
            *   **Провести "физикальный осмотр"**.
            *   **"Назначить" адекватные исследования** и "интерпретировать" их результаты.
            *   Сформулировать **дифференциальный диагноз**.
            *   Поставить **обоснованный окончательный (предварительный) диагноз**.
            *   Предложить **корректный и полный план ведения и лечения**.
            *   Эффективно **управлять временем**, если таймер включен.
            *   Демонстрировать **хорошие коммуникативные навыки**.
            *   Принимать решение о необходимости **консультации** взвешенно.
            """)
            st.subheader("⚖️ Система оценки:")
            st.markdown("По завершении сценария, LLM-преподаватель предоставит вам **детализированную оценку** по шкале от 0 до 10 баллов для каждого из следующих компонентов: сбор анамнеза, физикальный осмотр, диагностическое мышление, точность диагноза, план лечения, а также (если применимо) коммуникативные навыки. Вы также получите общую оценку, комментарий по тайм-менеджменту, комментарий по использованию консультаций, выделенные положительные моменты вашей работы и рекомендации по областям для улучшения.")
        st.markdown("---")
        st.info("👈 **Для начала работы, пожалуйста, настройте параметры в левой панели и либо сгенерируйте новый сценарий, либо выберите готовый на вкладке '📚 Готовые сценарии'. Просмотреть историю предыдущих сессий можно на вкладке '📈 История сессий'.**")

    with predef_scenarios_tab:
        st.subheader("📚 Выбор готового клинического сценария")
        if not SCENARIOS:
            st.warning("Список готовых сценариев пуст или не загружен. Вы можете сгенерировать сценарий с помощью LLM в панели слева.")
        else:
            filt_options_predef = ["Все уровни"] + DIFFICULTY_LEVELS
            if "difficulty_filter_predef" not in st.session_state:
                st.session_state.difficulty_filter_predef = filt_options_predef[0]

            selected_difficulty_filter_predef = st.selectbox(
                "Фильтр по уровню сложности:",
                filt_options_predef,
                key="difficulty_filter_predef",
                help="Выберите уровень сложности для отображения соответствующих сценариев."
            )

            filtered_scenarios_list = [
                s for s in SCENARIOS if isinstance(s, dict) and (selected_difficulty_filter_predef == "Все уровни" or s.get("difficulty_level_tag") == selected_difficulty_filter_predef)
            ]

            if not filtered_scenarios_list and selected_difficulty_filter_predef != "Все уровни":
                st.caption(f"Готовые сценарии с уровнем сложности '{selected_difficulty_filter_predef}' не найдены.")
            elif not filtered_scenarios_list:
                 st.caption("Список готовых сценариев пуст.")

            target_random_list = filtered_scenarios_list if selected_difficulty_filter_predef != "Все уровни" and filtered_scenarios_list else [s for s in SCENARIOS if isinstance(s, dict)]

            if st.button(f"🎲 Выбрать случайный сценарий {'из отфильтрованных' if selected_difficulty_filter_predef != 'Все уровни' and filtered_scenarios_list else 'из всех доступных'}", use_container_width=True, type="secondary"):
                if target_random_list:
                    chosen_scenario = random.choice(target_random_list)
                    initialize_scenario(chosen_scenario, st.session_state.start_with_hints_checkbox)
                    st.session_state.already_offered_training_mode_for_this_eval = False
                    st.rerun()
                else:
                    st.warning("Нет доступных сценариев для случайного выбора.")

            st.markdown("---")
            if filtered_scenarios_list:
                st.markdown(f"**Доступно сценариев (с учетом фильтра): {len(filtered_scenarios_list)}**")
                num_cols = 3
                scenario_cols = st.columns(num_cols)
                for idx, item_scenario in enumerate(filtered_scenarios_list):
                    col_to_use = scenario_cols[idx % num_cols]
                    if isinstance(item_scenario, dict) and "name" in item_scenario and "id" in item_scenario:
                        button_label = f"{item_scenario['name']} ({item_scenario.get('difficulty_level_tag', 'N/A')})"
                        if col_to_use.button(button_label, key=f"predef_scn_btn_{item_scenario['id']}_{idx}", use_container_width=True, help=f"Начать сценарий: {item_scenario['name']}"):
                            initialize_scenario(item_scenario, st.session_state.start_with_hints_checkbox)
                            st.session_state.already_offered_training_mode_for_this_eval = False
                            st.rerun()
            elif SCENARIOS and selected_difficulty_filter_predef != "Все уровни":
                pass
            elif not SCENARIOS:
                 pass
    
    with history_tab_initial:
        st.subheader("📈 Ваша история пройденных сессий")
        if st.session_state.session_history:
            df_history_display = pd.DataFrame(st.session_state.session_history)
            df_history_display.rename(columns={
                "name": "Название сценария",
                "score": "Итоговая оценка",
                "difficulty": "Уровень сложности",
                "date": "Дата и время",
                "time_taken": "Затраченное время",
                "timer_active": "Таймер был активен",
                "consultations_used": "Использовано консультаций"
            }, inplace=True, errors='ignore') 

            expected_cols = ["Название сценария", "Итоговая оценка", "Уровень сложности", "Дата и время", "Затраченное время", "Таймер был активен", "Использовано консультаций"]
            
            for col_name in expected_cols:
                if col_name not in df_history_display.columns:
                    df_history_display[col_name] = "N/A"
            
            df_history_display_final = df_history_display[expected_cols] 
            st.dataframe(df_history_display_final, hide_index=True, use_container_width=True)

            if st.button("🗑️ Очистить историю сессий", key="clear_session_history_button_initial", help="Это действие удалит всю сохраненную историю сессий."):
                st.session_state.session_history = []; st.rerun()
        else:
            st.info("Ваша история сессий пока пуста. Завершите сценарий, чтобы он появился здесь.")
