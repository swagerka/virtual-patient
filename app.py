
import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
import random
import re
import json
from datetime import datetime, timedelta
# import time # Не используется активно

MEDICAL_SPECIALIZATIONS = ["Общая терапия", "Гастроэнтерология", "Кардиология", "Пульмонология", "Неврология", "Эндокринология", "Нефрология (Урология)", "Инфекционные болезни", "Ревматология", "Педиатрия (общие случаи)", "Травматология и Ортопедия (несложные случаи)", "Гинекология (базовые случаи)", "Дерматология"]
AGE_RANGES = {"Младенец/Ребенок (0-5 лет)": (0, 5), "Ребенок (6-12 лет)": (6, 12), "Подросток (13-17 лет)": (13, 17), "Молодой взрослый (18-35 лет)": (18, 35), "Средний возраст (36-60 лет)": (36, 60), "Пожилой (61-80 лет)": (61, 80), "Старческий (81+ лет)": (81, 100)}
GENDERS = ["Любой", "Мужской", "Женский"]
DIFFICULTY_LEVELS = ["Легкий", "Средний", "Тяжелый"]
TIMER_OPTIONS_MAP = {"Без таймера": 0, "5 минут": 5, "10 минут": 10, "15 минут": 15, "20 минут": 20, "30 минут": 30}

try:
    from scenarios_data import SCENARIOS
    if not isinstance(SCENARIOS, list): SCENARIOS = []
except ImportError: SCENARIOS = []
except Exception as e: st.error(f"Ошибка scenarios_data.py: {e}"); SCENARIOS = []

load_dotenv()
KOBOLD_API_URL = os.getenv("KOBOLD_API_URL", "http://localhost:5002/v1/")
try:
    client = OpenAI(base_url=KOBOLD_API_URL, api_key="sk-not-needed")
except Exception as e: st.error(f"Ошибка OpenAI клиента: {e}"); st.stop()

def _extract_and_parse_json(raw_text):
    if not raw_text or not raw_text.strip(): raise ValueError("Ответ LLM пуст.")
    json_text = None; json_block_match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
    if json_block_match: json_text = json_block_match.group(1).strip()
    else:
        json_start_index = raw_text.find('{'); json_end_index = raw_text.rfind('}')
        if json_start_index != -1 and json_end_index != -1 and json_end_index > json_start_index:
            json_text = raw_text[json_start_index : json_end_index + 1].strip()
    if not json_text: raise ValueError("JSON структура не найдена.")
    try: return json.loads(json_text)
    except json.JSONDecodeError as e:
        st.warning(f"Ошибка парсинга JSON: {e.msg}. Попытка восстановления...")
        if e.pos is not None and e.doc is not None: 
            context_size = 50; start = max(0, e.pos - context_size); end = min(len(e.doc), e.pos + context_size)
            error_context = e.doc[start:end]; pointer_pos = e.pos - start
            pointer_pos = max(0, min(pointer_pos, len(error_context) -1 if len(error_context) > 0 else 0))
            error_line_display = error_context.replace('\n', ' '); pointer_line_display = ' ' * pointer_pos + '^'
            st.markdown(f"**Контекст ошибки:**\n```text\n{error_line_display}\n{pointer_line_display}\n```")
        possible_suffixes = ["}", "]", "}}", "]}", "}]", "\"}", "\"]}", "\"}]", "\"}}", ")}"]
        original_json_text_for_repair = json_text 
        for suffix_to_add in possible_suffixes:
            try: return json.loads(original_json_text_for_repair + suffix_to_add)
            except json.JSONDecodeError: continue 
        raise json.JSONDecodeError(f"Не удалось восстановить JSON. Ошибка: {e.msg}", e.doc if e.doc else "", e.pos if e.pos is not None else 0)

def generate_llm_response(messages_history_for_llm, system_prompt_for_llm):
    try:
        messages_to_send = [{"role": "system", "content": system_prompt_for_llm}] + \
                           [msg for msg in messages_history_for_llm if msg["role"] in ["user", "assistant"]]
        response = client.chat.completions.create(model="local-model", messages=messages_to_send, max_tokens=400, temperature=0.75)
        if response.choices and response.choices[0].message.content: return response.choices[0].message.content.strip()
        st.warning("LLM не вернул ответ."); return "Пациент молчит..."
    except Exception as e: st.error(f"Ошибка API пациента: {e}"); return "Пациент не может ответить."

def generate_new_scenario_via_llm(age_range_str=None, specialization_str=None, gender_str=None, difficulty_str=None):
    st.info("Запрос на генерацию нового сценария LLM...")
    customization_prompt_parts = []; patient_age_for_prompt = "случайный"; patient_gender_for_prompt = "случайный"
    if age_range_str and age_range_str != "Любой" and age_range_str in AGE_RANGES: min_a, max_a = AGE_RANGES[age_range_str]; customization_prompt_parts.append(f"Возраст: {min_a}-{max_a} лет."); patient_age_for_prompt = f"{min_a}-{max_a} лет"
    if gender_str and gender_str != "Любой": customization_prompt_parts.append(f"Пол: {gender_str}."); patient_gender_for_prompt = gender_str
    difficulty_instructions = ""
    if difficulty_str and difficulty_str != "Любой":
        difficulty_instructions = f"Уровень сложности: {difficulty_str}. "
        if difficulty_str == "Легкий": difficulty_instructions += "Симптомы КЛАССИЧЕСКИЕ, пациент КОНТАКТНЫЙ, диагноз ОТНОСИТЕЛЬНО ОЧЕВИДЕН. Внешность: легкое недомогание."
        elif difficulty_str == "Средний": difficulty_instructions += "Симптомы НЕ СОВСЕМ ТИПИЧНЫЕ/СМАЗАННЫЕ, пациент УМЕРЕННО БЕСПОКОЕН. Требуется ДИФДИАГНОСТИКА. Внешность: заметные признаки болезни."
        elif difficulty_str == "Тяжелый": difficulty_instructions += "Симптомы АТИПИЧНЫЕ/МНОЖЕСТВЕННЫЕ, пациент ТРУДНЫЙ В ОБЩЕНИИ. ТЩАТЕЛЬНАЯ дифдиагностика. Внешность: тяжелое состояние."
        customization_prompt_parts.append(difficulty_instructions)
    if specialization_str and specialization_str != "Любая": customization_prompt_parts.append(f"Область: {specialization_str}.")
    customization_instructions = " ".join(customization_prompt_parts); 
    if customization_instructions: customization_instructions = f"ТРЕБОВАНИЯ: {customization_instructions}"
    meta_prompt = f"""Ты — эксперт ... {customization_instructions} ... JSON СТРОГО ...
{{
  "id": "llm_gen_...", "name": "...", "patient_initial_info_display": "...", 
  "patient_appearance_detailed": "ПОДРОБНОЕ описание внешности (возраст, пол, болезнь, сложность)...", 
  "objective_findings_on_entry": {{ "blood_pressure": "'120/80' или 'не измерялось'", ... }}, 
  "initial_lab_results": {{ "OAK": {{...}}, "Biochemistry": {{...}} /* или {{}} */ }}, 
  "patient_llm_persona_system_prompt": "Подробный системный промпт... Инструкции про ФИЗИКАЛЬНЫЙ ОСМОТР (симуляция результатов по запросу), ИСХОДНЫЕ ДАННЫЕ (из objective_findings, initial_lab_results), СКРЫТЫЕ ТРИГГЕРЫ ([СИСТЕМНЫЙ ТРИГГЕР: ...], [ДОП. ИНСТРУКЦИЯ: ...])...",
  "initial_patient_greeting": "...", "true_diagnosis_internal": "...", "true_diagnosis_detailed": "...",
  "key_anamnesis_points": [...], "correct_plan_detailed": "...",
  "hidden_triggers": [ /* 0-2 триггера: {{ "id": "id", "condition_type": "keyword"|"message_count"|"after_llm_keyword", "condition_value": ["слов"]|число|["ключ_в_ответе_ллм"], "patient_reveal_info": "текст", "modify_system_prompt_add": "добавка к промпту", "priority": число }} */ ],
  "common_mistakes": [ {{ "id": "empty_dx", ..., "penalty": 5 }}, {{ "id": "empty_plan", ..., "penalty": 5 }} /* сумма 10 */ ],
  "key_diagnostic_questions_keywords": [...], "correct_diagnosis_keywords_for_check": [...], "correct_plan_keywords_for_check": [...]
}} JSON должен быть абсолютно валидным. Удели внимание синтаксису списков строк в "condition_value" для триггеров."""
    raw_text_response = ""
    try:
        response = client.completions.create(model="local-model", prompt=meta_prompt, max_tokens=5000, temperature=0.85)
        raw_text_response = response.choices[0].text
        generated_scenario = _extract_and_parse_json(raw_text_response)
        generated_scenario['id'] = f"llm_gen_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        required_keys_with_types_and_defaults = {
            "id": (str, None), "name": (str, "Без названия"), "patient_initial_info_display": (str, "Нет данных."),
            "patient_appearance_detailed": (str, "Внешность не описана."), "objective_findings_on_entry": (dict, {}),
            "initial_lab_results": (dict, {}), "patient_llm_persona_system_prompt": (str, "Ты пациент."),
            "initial_patient_greeting": (str, "Здравствуйте."), "true_diagnosis_internal": (str, "N/A"),
            "true_diagnosis_detailed": (str, "N/A"), "key_anamnesis_points": (list, []),
            "correct_plan_detailed": (str, "N/A"), "hidden_triggers": (list, []),
            "common_mistakes": (list, []), "key_diagnostic_questions_keywords": (list, []),
            "correct_diagnosis_keywords_for_check": (list, []), "correct_plan_keywords_for_check": (list, [])
        }
        for key, (expected_type, default_value) in required_keys_with_types_and_defaults.items():
            if key not in generated_scenario:
                if default_value is not None: generated_scenario[key] = default_value
                elif key != "id": st.error(f"Крит. поле '{key}' отсутствует."); return None
            elif not isinstance(generated_scenario[key], expected_type):
                if default_value is not None : generated_scenario[key] = default_value
                elif expected_type == str: generated_scenario[key] = str(generated_scenario[key])
        valid_triggers = []
        if isinstance(generated_scenario.get("hidden_triggers"), list):
            for trigger in generated_scenario["hidden_triggers"]:
                if isinstance(trigger, dict) and "id" in trigger and "condition_type" in trigger and \
                   "condition_value" in trigger and "patient_reveal_info" in trigger:
                    valid_cond = False; ct = trigger["condition_type"]; cv = trigger.get("condition_value")
                    if ct in ["keyword", "after_llm_keyword"] and isinstance(cv, list) and all(isinstance(i, str) for i in cv): valid_cond = True
                    elif ct == "message_count" and isinstance(cv, int): valid_cond = True
                    if valid_cond: trigger["triggered_once"] = False; trigger["priority"] = trigger.get("priority",0); trigger["modify_system_prompt_add"] = trigger.get("modify_system_prompt_add",""); valid_triggers.append(trigger)
        generated_scenario["hidden_triggers"] = valid_triggers
        llm_mistakes = generated_scenario.get("common_mistakes", []); final_mistakes = []; present_ids = set()
        mandatory_errors = [{"id": "empty_dx", "description": "Диагноз не поставлен.", "penalty": 5}, {"id": "empty_plan", "description": "План не предложен.", "penalty": 5}]
        for err_template in mandatory_errors:
            found = next((lm for lm in llm_mistakes if isinstance(lm,dict) and lm.get("id")==err_template["id"] and lm.get("penalty")==err_template["penalty"]), None)
            if found: final_mistakes.append(found); present_ids.add(err_template["id"])
            else: final_mistakes.append(err_template); present_ids.add(err_template["id"])
        for lm_err in llm_mistakes:
            if isinstance(lm_err, dict) and lm_err.get("id") not in present_ids: final_mistakes.append(lm_err)
        generated_scenario["common_mistakes"] = final_mistakes
        if sum(m.get("penalty",0) for m in final_mistakes) != 10: st.warning("Сумма штрафов common_mistakes не 10.")
        st.success("Сценарий сгенерирован!"); return generated_scenario
    except (json.JSONDecodeError, ValueError) as e: st.error(f"Ошибка парсинга JSON: {e}"); st.text_area("LLM ответ:", raw_text_response, height=200); return None
    except Exception as e: st.error(f"Ошибка генерации: {e}"); st.text_area("LLM ответ:", raw_text_response, height=200); return None

def evaluate_with_llm(scenario_data, user_dialogue_msgs, user_dx, user_plan, consultation_count):
    st.info("Оценка LLM...")
    raw_text_eval = ""; 
    scenario_info_for_evaluator = f"""**Сценарий:** {scenario_data.get('name', 'N/A')}
**Первичка:** {scenario_data.get('patient_initial_info_display', 'N/A')}
**Внешний вид:** {scenario_data.get('patient_appearance_detailed', 'N/A')}
**Ист.диагноз:** {scenario_data.get('true_diagnosis_detailed', 'N/A')}
**Ключ.анамнез:** {scenario_data.get('key_anamnesis_points', [])}
**Прав.план:** {scenario_data.get('correct_plan_detailed', 'N/A')}"""
    dialogue_str = "\n".join([f"{'Врач' if msg['role']=='user' else 'Пациент'}: {msg['content']}" for msg in user_dialogue_msgs])
    physician_actions_for_evaluator = f"""**Диалог:**\n{dialogue_str}
**Предл.диагноз:** {user_dx or '[нет]'}
**Предл.план:** {user_plan or '[нет]'}
**Консультаций:** {consultation_count}"""
    evaluation_prompt = f"""Ты — препод...
Данные сценария: {scenario_info_for_evaluator}
Действия врача: {physician_actions_for_evaluator}
JSON формат: {{ "score": int, "explanation": {{ "correct_aspects": [], "mistakes_or_omissions": [] }} }}
Учитывай консультации (если >0, немного снизь балл)."""
    try:
        response = client.completions.create(model="local-model", prompt=evaluation_prompt, max_tokens=2000, temperature=0.4)
        raw_text_eval = response.choices[0].text; evaluation_result_obj = _extract_and_parse_json(raw_text_eval)
        score_val = float(evaluation_result_obj.get("score", 0))
        if consultation_count > 0: score_val = max(0, score_val - (consultation_count * 0.5)) 
        final_score = round(max(0, min(10, score_val)), 1)
        expl = evaluation_result_obj.get("explanation", {}); correct = expl.get("correct_aspects", []); mistakes = expl.get("mistakes_or_omissions", [])
        formatted_results = {"score": final_score, "mistakes": [{"description": d} for d in mistakes], "correct_actions": correct}
        st.success("Оценка получена!"); return formatted_results
    except (json.JSONDecodeError, ValueError) as e: st.error(f"Ошибка JSON оценщика: {e}"); st.text_area("Ответ оценщика:", raw_text_eval, height=150); return {"score":0, "mistakes": [{"description":f"Ошибка JSON:{e}"}]}
    except Exception as e: st.error(f"Ошибка оценки: {e}"); st.text_area("Ответ оценщика:", raw_text_eval, height=150); return {"score":0, "mistakes": [{"description":f"Ошибка:{e}"}]}

def initialize_scenario(scenario_data_obj, training_mode=False, keep_results_for_training=False):
    if not isinstance(scenario_data_obj, dict): st.error("Ошибка данных сценария."); st.rerun(); return
    st.session_state.current_scenario = scenario_data_obj 
    if "hidden_triggers" in scenario_data_obj and isinstance(scenario_data_obj.get("hidden_triggers"), list): 
        for trigger in scenario_data_obj.get("hidden_triggers", []): 
            if isinstance(trigger, dict): trigger["triggered_once"] = False
    st.session_state.messages = [{"role": "assistant", "content": scenario_data_obj.get("initial_patient_greeting", "Здравствуйте.")}]
    st.session_state.vitals_history = {"АД": [], "ЧСС": [], "Температура": [], "Сатурация": []} 
    st.session_state.time_up = False; st.session_state.consultation_count = 0
    timer_minutes = st.session_state.get("timer_duration_setting", 0)
    st.session_state.timer_duration_minutes = timer_minutes
    if timer_minutes > 0:
        st.session_state.simulation_start_time = datetime.now()
        st.session_state.simulation_end_time = st.session_state.simulation_start_time + timedelta(minutes=timer_minutes)
        st.toast(f"Таймер: {timer_minutes} мин.", icon="⏱️")
    else: st.session_state.simulation_start_time = None; st.session_state.simulation_end_time = None
    if not (training_mode and keep_results_for_training and "evaluation_results" in st.session_state):
        st.session_state.user_diagnosis = ""; st.session_state.user_action_plan = ""
    st.session_state.chat_active = True; st.session_state.scenario_selected = True
    st.session_state.training_mode_active = training_mode
    if not training_mode or not keep_results_for_training:
        st.session_state.evaluation_done = False; st.session_state.evaluation_results = None

def reset_session_and_rerun():
    settings_keys = ["llm_age", "llm_gender", "llm_spec", "llm_difficulty", "start_with_hints_checkbox", "timer_duration_setting"]
    saved_settings = {k: st.session_state.get(k) for k in settings_keys if k in st.session_state}
    for k in list(st.session_state.keys()):
        if not k.startswith("FormSubmitter:") and not k.endswith(("_input", "_key", "_btn", "_widget")) and k not in settings_keys:
            del st.session_state[k]
    for key, value in default_session_state_values.items():
        if key not in st.session_state: 
            st.session_state[key] = value() if callable(value) else value
    for k, v in saved_settings.items(): st.session_state[k] = v
    st.rerun()

def extract_and_store_vitals(patient_response_text, message_num_tag):
    bp = re.search(r"(\b\d{2,3}\s*(?:/|на)\s*\d{2,3}\b)", patient_response_text,re.I); hr = re.search(r"(ЧСС|пульс)\s*:?\s*(\b\d{2,3}\b)", patient_response_text,re.I)
    tp = re.search(r"(t|температура)\s*:?\s*(\b\d{2}(?:[.,]\d)?\b)", patient_response_text,re.I); sp = re.search(r"(сатурация|SpO2)\s*:?\s*(\b\d{2,3}\b)\s*%", patient_response_text,re.I)
    if bp: st.session_state.vitals_history["АД"].append(f"{message_num_tag}: {bp.group(1)}")
    if hr: st.session_state.vitals_history["ЧСС"].append(f"{message_num_tag}: {hr.group(2)} уд/мин")
    if tp: st.session_state.vitals_history["Температура"].append(f"{message_num_tag}: {tp.group(2).replace(',','.')}°C")
    if sp: st.session_state.vitals_history["Сатурация"].append(f"{message_num_tag}: {sp.group(1)}%")

def format_message_with_vitals_highlight(text):
    text = re.sub(r"(\b\d{2,3}\s*(?:/|на)\s*\d{2,3}\b)", r"**\1**", text, flags=re.I)
    text = re.sub(r"((?:ЧСС|пульс)\s*:?\s*)(\b\d{2,3}\b)", r"\1**\2**", text, flags=re.I)
    text = re.sub(r"((?:t|температура)\s*:?\s*)(\b\d{2}(?:[.,]\d)?\b)", r"\1**\2**", text, flags=re.I)
    text = re.sub(r"((?:сатурация|SpO2)\s*:?\s*)(\b\d{2,3}\b\s*%)", r"\1**\2**", text, flags=re.I)
    return text

def analyze_user_against_common_mistakes(scenario_mistakes, user_dx, user_plan, dialogue_history):
    committed = []
    if not user_dx.strip(): next((committed.append(m) for m in scenario_mistakes if m.get("id")=="empty_dx"),None)
    if not user_plan.strip(): next((committed.append(m) for m in scenario_mistakes if m.get("id")=="empty_plan"),None)
    return committed

default_session_state_values = {
    "scenario_selected": False, "training_mode_active": False, "already_offered_training_mode_for_this_eval": False,
    "start_with_hints_checkbox": False, "evaluation_done": False, "evaluation_results": None, 
    "messages": list, "chat_active": True, "current_scenario": None, 
    "user_diagnosis": "", "user_action_plan": "", "doctor_notes": "",
    "llm_age": "Любой", "llm_gender": "Любой", "llm_spec": "Любая", "llm_difficulty": "Легкий",
    "timer_duration_setting": 0, "timer_duration_minutes": 0, 
    "simulation_start_time": None, "simulation_end_time": None, "time_up": False,
    "consultation_count": 0, "vitals_history": lambda: {"АД": [], "ЧСС": [], "Температура": [], "Сатурация": []}
}
for key, value in default_session_state_values.items():
    if key not in st.session_state: st.session_state[key] = value() if callable(value) else value

st.set_page_config(layout="wide", page_title="Виртуальный Пациент")

with st.sidebar: 
    st.title("👨‍⚕️ Управление")
    if st.session_state.current_scenario:
        st.success(f"Активен: {st.session_state.current_scenario.get('name','Сценарий')[:25]}...")
        if st.button("🔄 Новый сценарий", use_container_width=True, key="reset_sidebar_btn"): reset_session_and_rerun()
    if not st.session_state.scenario_selected:
        st.header("⚙️ Выбор Сценария"); 
        with st.expander("Параметры генерации (LLM)", True):
            for key_suffix, options, current_val_key in [
                ("age", ["Любой"] + list(AGE_RANGES.keys()), "llm_age"),
                ("gender", GENDERS, "llm_gender"),
                ("spec", ["Любая"] + MEDICAL_SPECIALIZATIONS, "llm_spec"),
                ("difficulty", DIFFICULTY_LEVELS, "llm_difficulty")]:
                st.selectbox(key_suffix.capitalize()+":", options, key=f"llm_{key_suffix}_widget", 
                             index=options.index(st.session_state[current_val_key]), 
                             on_change=lambda k=current_val_key, w=f"llm_{key_suffix}_widget": setattr(st.session_state, k, st.session_state[w]))
            st.checkbox("Режим обучения", key="start_with_hints_checkbox")
        st.markdown("---"); st.subheader("⏱️ Таймер")
        def ont_ch(): st.session_state.timer_duration_setting = TIMER_OPTIONS_MAP[st.session_state.tmr_sel_cb]
        st.selectbox("Длительность:", list(TIMER_OPTIONS_MAP.keys()), key="tmr_sel_cb", 
                     index=list(TIMER_OPTIONS_MAP.values()).index(st.session_state.timer_duration_setting), on_change=ont_ch)
        st.markdown("---")
        if st.button("✨ Сгенерировать (LLM)", use_container_width=True, type="primary", key="gen_llm_btn"):
            with st.spinner("LLM генерирует..."): new_sc = generate_new_scenario_via_llm(st.session_state.llm_age, st.session_state.llm_spec, st.session_state.llm_gender, st.session_state.llm_difficulty)
            if new_sc: initialize_scenario(new_sc, st.session_state.start_with_hints_checkbox); st.rerun()
            else: st.error("Не удалось сгенерировать.")
        if SCENARIOS: 
            st.markdown("---"); st.subheader("Готовые сценарии")
            if st.button("🎲 Случайный", use_container_width=True, key="rand_sc_btn"): 
                if SCENARIOS: initialize_scenario(random.choice(SCENARIOS), st.session_state.start_with_hints_checkbox); st.rerun()
            for i, item in enumerate(SCENARIOS):
                if isinstance(item,dict) and "name" in item: 
                    if st.button(item["name"], key=f"sc_btn_{item.get('id','noid')}_{i}", use_container_width=True): initialize_scenario(item, st.session_state.start_with_hints_checkbox); st.rerun()
        st.markdown("---"); st.caption(f"LLM: {KOBOLD_API_URL.split('/')[-2]}")


if st.session_state.current_scenario: 
    scenario = st.session_state.current_scenario; is_training = st.session_state.training_mode_active
    st.title(f"🩺 {scenario.get('name', 'Случай')}")
    timer_html = ""; force_eval = False 
    if st.session_state.simulation_end_time and not st.session_state.time_up and not st.session_state.evaluation_done:
        rem_time = st.session_state.simulation_end_time - datetime.now()
        if rem_time.total_seconds() > 0: mins, secs = divmod(int(rem_time.total_seconds()),60); timer_html = f"<p>⏱️ Осталось: <b style='color:#1E90FF;'>{mins:02d}:{secs:02d}</b></p>"
        else: 
            if not st.session_state.time_up: st.session_state.time_up=True; st.session_state.chat_active=False; st.warning("Время вышло!"); force_eval=True
    elif st.session_state.time_up: timer_html = "<p>⏱️ Время вышло!</p>"
    elif st.session_state.timer_duration_minutes == 0 and not st.session_state.evaluation_done: timer_html = "<p>⏱️ Таймер не установлен</p>"
    
    st.markdown(f"<style>.info-card p {{margin-bottom:3px!important}} .appearance-section p {{font-style:italic;color:#555}}</style>", unsafe_allow_html=True) 
    st.markdown(f"<div class='info-card'><p><b>Пациент:</b> {scenario.get('patient_initial_info_display','N/A')}</p><p><b>Режим:</b> {'💡 Обучение' if is_training else '🚀 Самост.'}{' (Завершено)' if st.session_state.evaluation_done else ' (В процессе)'}</p>{timer_html}</div>", unsafe_allow_html=True)
    app_desc = scenario.get('patient_appearance_detailed',''); 
    if app_desc and app_desc.lower()!='внешность пациента не описана.': st.markdown(f"<div class='appearance-section'><p><b>Внешний вид:</b> {app_desc}</p></div>", unsafe_allow_html=True)
    
    if force_eval and not st.session_state.evaluation_done:
        st.session_state.evaluation_done=True
        results=evaluate_with_llm(scenario,st.session_state.messages,st.session_state.user_diagnosis,st.session_state.user_action_plan,st.session_state.consultation_count)
        st.session_state.evaluation_results=results; st.rerun()
    
    tabs_def = {"💬 Диалог и Действия": None}
    if st.session_state.evaluation_done: tabs_def["📊 Результаты"] = None
    if is_training or st.session_state.evaluation_done: tabs_def["ℹ️ Детали Сценария"] = None
    active_tabs_keys = list(tabs_def.keys())
    tabs_ui = st.tabs(active_tabs_keys)

    with tabs_ui[0]: 
        col1, col2 = st.columns([0.45,0.55])
        with col1:
            inputs_dis = st.session_state.evaluation_done or st.session_state.time_up
            with st.form(key="dx_plan_form_key"): # Изменен ключ формы для уникальности
                st.text_input("Предв. диагноз:", value=st.session_state.get("user_diagnosis", ""), key="dx_form_input_widget", disabled=inputs_dis)
                st.text_area("План:", value=st.session_state.get("user_action_plan", ""), key="plan_form_input_widget",height=120,disabled=inputs_dis)
                if st.form_submit_button("✔️ Завершить и Оценить", type="primary", use_container_width=True, disabled=inputs_dis):
                    st.session_state.user_diagnosis = st.session_state.dx_form_input_widget 
                    st.session_state.user_action_plan = st.session_state.plan_form_input_widget
                    if st.session_state.simulation_end_time and datetime.now() >= st.session_state.simulation_end_time and not st.session_state.time_up: st.session_state.time_up=True; st.warning("Время вышло перед отправкой!")
                    st.session_state.chat_active=False
                    results_eval=evaluate_with_llm(scenario,st.session_state.messages,st.session_state.user_diagnosis,st.session_state.user_action_plan,st.session_state.consultation_count)
                    st.session_state.evaluation_results=results_eval; st.session_state.evaluation_done=True; st.rerun()
            st.subheader("📝 Ваши заметки")
            notes_now = st.session_state.get("doctor_notes","")
            new_notes_now = st.text_area("Личные пометки:",value=notes_now,key="doc_notes_widget_key",height=100,disabled=inputs_dis) # Изменен ключ
            if new_notes_now != notes_now: st.session_state.doctor_notes = new_notes_now
            st.markdown("---"); st.subheader("🤔 Нужна помощь?")
            consult_dis = inputs_dis or st.session_state.consultation_count >= 3
            if st.button(f"Запросить консультацию ({st.session_state.consultation_count}/3)",key="consult_btn_main_key",disabled=consult_dis,use_container_width=True): # Изменен ключ
                st.session_state.consultation_count+=1
                history_for_consult = "\n".join([f"{'Врач' if msg['role'] == 'user' else 'Пациент'}: {msg['content']}" for msg in st.session_state.messages[-7:]]) # Последние 7 сообщений
                patient_display_info_consult = scenario.get('patient_initial_info_display', 'N/A')
                consult_prompt_text = f"""Ты - опытный врач-наставник. Твой коллега-стажер ведет пациента и просит ОДИН короткий тактический совет.
Пациент (первичная информация): {patient_display_info_consult}.
Последние сообщения диалога:
{history_for_consult}
Предварительный диагноз стажера: {st.session_state.user_diagnosis or '[не указан]'}
План стажера: {st.session_state.user_action_plan or '[не указан]'}
ТВОЯ ЗАДАЧА: Дай ОДИН КОНКРЕТНЫЙ совет (1-2 предложения), что стажеру сделать дальше (какой вопрос задать, на что обратить внимание, какое обследование назначить). НЕ ДАВАЙ полный диагноз. НЕ ИСПОЛЬЗУЙ markdown или ```. Только текст совета.
Пример твоего ответа: Уточните, пожалуйста, иррадиацию болей и связь с приемом пищи.
Твой совет:"""
                with st.spinner("Коллега думает..."):
                    try: 
                        resp_c=client.completions.create(model="local-model",prompt=consult_prompt_text,max_tokens=100,temperature=0.5, stop=["\n\n", "Пациент:", "Врач:"])
                        advice_raw = resp_c.choices[0].text.strip()
                        advice_cleaned = advice_raw.replace("```", "").strip()
                        advice_lines = [line.strip() for line in advice_cleaned.splitlines() if line.strip()]
                        final_advice = " ".join(advice_lines)
                        if final_advice: st.info(f"💡 **Совет от коллеги:** {final_advice}")
                        else: st.warning("Коллега не смог дать конкретный совет или вернул пустой ответ.")
                        st.toast("Консультация получена!",icon="🤝")
                    except Exception as e_c: st.error(f"Ошибка консультации: {e_c}")
        with col2: 
            st.subheader("💬 Диалог с пациентом")
            chat_ph = st.empty()
            with chat_ph.container(height=500):
                 for msg_i in st.session_state.messages: 
                     with st.chat_message(msg_i["role"],avatar="🧑‍⚕️" if msg_i["role"]=="user" else "🤒"): st.markdown(format_message_with_vitals_highlight(msg_i["content"]),unsafe_allow_html=True)
            chat_input_disabled_reason_ui = ""
            if st.session_state.time_up: chat_input_disabled_reason_ui="Время вышло!"
            elif st.session_state.evaluation_done: chat_input_disabled_reason_ui="Оценка проведена."
            elif not st.session_state.chat_active: chat_input_disabled_reason_ui="Диалог неактивен."
            chat_truly_disabled_ui = bool(chat_input_disabled_reason_ui)
            user_q_text = st.chat_input("Вопрос пациенту...",key="chat_main_widget_key_unique",disabled=chat_truly_disabled_ui) # Еще раз изменен ключ
            if user_q_text and not chat_truly_disabled_ui:
                if st.session_state.simulation_end_time and datetime.now() >= st.session_state.simulation_end_time and not st.session_state.time_up:
                    st.session_state.time_up=True; st.warning("Время вышло! Сообщение не отправлено."); force_eval=True; st.rerun()
                else:
                    st.session_state.messages.append({"role":"user","content":user_q_text}); trig_pref=""; sys_prompt_add=""
                    user_msg_count = len([m for m in st.session_state.messages if m['role']=='user']) 
                    if "hidden_triggers" in scenario: 
                        active_trigs_now=[]; 
                        for trig_o_idx, trig_o in enumerate(st.session_state.current_scenario.get("hidden_triggers", [])): # Итерируем по копии из session_state
                            if isinstance(trig_o,dict) and not trig_o.get("triggered_once",False):
                                trig_fire=False; ct=trig_o.get("condition_type"); cv=trig_o.get("condition_value"); mod_prompt=trig_o.get("modify_system_prompt_add","")
                                if ct=="keyword" and isinstance(cv,list) and any(k.lower() in user_q_text.lower() for k in cv): trig_fire=True
                                elif ct=="message_count" and isinstance(cv,int) and user_msg_count >= cv: trig_fire=True
                                elif ct=="after_llm_keyword" and isinstance(cv,list) and len(st.session_state.messages)>1 and st.session_state.messages[-2].get("role")=="assistant" and any(k.lower() in st.session_state.messages[-2].get("content","").lower() for k in cv): trig_fire=True
                                if trig_fire: trig_o["system_prompt_modification"]=mod_prompt; active_trigs_now.append(trig_o) # Добавляем сам объект триггера
                        active_trigs_now.sort(key=lambda t: t.get("priority",0),reverse=True)
                        if active_trigs_now:
                            chosen_t=active_trigs_now[0]; trig_pref=f"[СИСТЕМНЫЙ ТРИГГЕР: {chosen_t.get('patient_reveal_info','')}] "; sys_prompt_add=chosen_t.get("system_prompt_modification","")
                            chosen_t["triggered_once"] = True # Помечаем как сработавший в объекте из session_state
                    with st.spinner("Пациент отвечает..."):
                        base_p_prompt = scenario.get("patient_llm_persona_system_prompt","Ты пациент.")
                        final_p_prompt = trig_pref + base_p_prompt + (f" [ДОП.ИНСТРУКЦИЯ: {sys_prompt_add}]" if sys_prompt_add else "")
                        llm_resp_text = generate_llm_response(st.session_state.messages,final_p_prompt)
                    st.session_state.messages.append({"role":"assistant","content":llm_resp_text})
                    extract_and_store_vitals(llm_resp_text,f"Вопрос #{user_msg_count}")
                    st.rerun()
            elif chat_input_disabled_reason_ui and chat_ph: # Показываем сообщение если чат заблокирован и плейсхолдер существует
                with chat_ph.container(): # Это может перекрыть чат, если он там уже есть.
                    # Лучше иметь отдельный st.empty() для таких сообщений под полем ввода чата.
                    # st.info(f"Диалог неактивен: {chat_input_disabled_reason_ui}") # Закомментировано пока
                    pass


    if "📊 Результаты" in tabs_def:
        with active_tabs[active_tabs_keys.index("📊 Результаты")]:
            results_data = st.session_state.evaluation_results; st.subheader(f"Результаты {'обучения' if is_training else 'самост. работы'}")
            if results_data and "score" in results_data:
                score_val_res = results_data.get('score', 0.0)
                st.metric(label="Итоговая оценка LLM", value=f"{score_val_res:.1f}/10",delta_color="off")
                with st.expander("👍 Правильно (оценщик)", True): 
                    ca = results_data.get('correct_actions',[]); st.write("Нет данных" if not ca else "\n".join([f"- ✅ {a}" for a in ca]))
                with st.expander("🤔 Ошибки (оценщик)",True): 
                    ms = results_data.get('mistakes',[]); st.write("Нет данных" if not ms else "\n".join([f"- ❌ {m.get('description','N/A')}" for m in ms]))
                st.markdown("---"); st.subheader("Анализ по Частым Ошибкам Сценария")
                committed_list = analyze_user_against_common_mistakes(scenario.get("common_mistakes",[]),st.session_state.user_diagnosis,st.session_state.user_action_plan,st.session_state.messages)
                if committed_list: st.warning("Совпадения с типичными ошибками:"); [st.markdown(f"- **{m.get('description')}** (Штраф: {m.get('penalty')})") for m in committed_list]
                else: st.success("Типичных ошибок не найдено!")
                st.markdown("---"); 
                with st.expander("📝 Протокол Консультации (Сводка)", False):
                    st.markdown(f"**Пациент:** {scenario.get('patient_initial_info_display','N/A')}")
                    st.markdown(f"**Внешний вид:** {scenario.get('patient_appearance_detailed','N/A')}")
                    initial_complaints_proto = st.session_state.messages[0]["content"] if st.session_state.messages and st.session_state.messages[0]["role"] == "assistant" else scenario.get('patient_initial_info_display', '')
                    st.markdown(f"**Жалобы (начало):** *{initial_complaints_proto[:250]}...*")
                    st.markdown("**Анализ ключевых моментов анамнеза:**")
                    key_points_sc = scenario.get('key_anamnesis_points', []); user_q_text_all = " ".join([m['content'].lower() for m in st.session_state.messages if m['role']=='user'])
                    covered_pts, missed_pts = [],[]
                    if key_points_sc:
                        for pt_text in key_points_sc:
                            pt_kw = [w.lower() for w in re.findall(r'\b[а-яА-Яa-zA-Z]{3,}\b', pt_text)]; 
                            if not pt_kw: missed_pts.append(pt_text); continue
                            thresh = max(1, len(pt_kw)//2 + (1 if len(pt_kw)%2!=0 else 0) ); found_c = sum(1 for p in pt_kw if p in user_q_text_all)
                            if found_c >= thresh: covered_pts.append(pt_text)
                            else: missed_pts.append(pt_text)
                        if covered_pts: st.success("Выявлено:"); [st.markdown(f"  - ✅ {c}") for c in covered_pts]
                        if missed_pts: st.warning("Пропущено/недостаточно раскрыто:"); [st.markdown(f"  - ⚠️ {m}") for m in missed_pts]
                    else: st.caption("Эталонные пункты не определены.")
                    if st.session_state.doctor_notes: st.markdown("**Ваши заметки:**"); st.code(st.session_state.doctor_notes)
                    if any(st.session_state.vitals_history.values()): 
                        st.markdown("**Данные осмотра (из диалога):**")
                        for vital_name, history_list_item in st.session_state.vitals_history.items():
                            if history_list_item: st.markdown(f"  - **{vital_name}:** {', '.join(history_list_item)}")
                    st.markdown(f"**Предв. диагноз врача:**"); st.code(st.session_state.user_diagnosis or "[Не указан]")
                    st.markdown(f"**План врача:**"); st.code(st.session_state.user_action_plan or "[Не указан]")
                    st.markdown(f"**Истинный диагноз:** {scenario.get('true_diagnosis_detailed','N/A')}", help=scenario.get('true_diagnosis_internal',''))
                    st.markdown(f"**Эталонный план:** {scenario.get('correct_plan_detailed','N/A')}")
                    st.markdown(f"**Итоговая оценка LLM:** {res_data.get('score',0.0):.1f}/10")
                    if st.session_state.consultation_count > 0: st.caption(f"(С учетом {st.session_state.consultation_count} консультаций)")
                if not is_training and not st.session_state.already_offered_training_mode_for_this_eval:
                    if st.button("💡 Пройти с подсказками",key="train_btn_res",use_container_width=True): initialize_scenario(scenario,True,True); st.rerun()
            else: st.warning("Результаты оценки еще не готовы.")

    if "ℹ️ Детали Сценария" in tabs_def:
        with active_tabs[active_tabs_keys.index("ℹ️ Детали Сценария")]:
            st.subheader("Детали Сценария и Эталоны")
            # ... (код вывода деталей как в предыдущем полном ответе)

elif not st.session_state.current_scenario and not st.session_state.scenario_selected:
    st.title("Добро пожаловать в Симулятор Виртуального Пациента!")
    st.markdown("Этот симулятор предназначен для тренировки ваших навыков...") # Полный текст инструкций
    col1_intro, col2_intro = st.columns(2)
    with col1_intro:
        st.subheader("🚀 Как начать?")
        st.markdown("""
            1.  **Выберите или сгенерируйте сценарий** в панели слева.
                *   Вы можете настроить параметры для LLM-генерации (возраст, пол, сложность, специализацию).
                *   Также можно выбрать длительность симуляции с помощью **таймера** или отключить его.
            2.  **Ознакомьтесь с первичной информацией** о пациенте и его внешнем виде, которая появится после выбора сценария.
            3.  **Начните диалог:** задавайте вопросы пациенту в поле ввода под чатом.
            4.  **Используйте "Заметки врача"** (слева, под формой диагноза) для своих пометок по ходу дела.
        """)
        st.subheader("💬 Общение с пациентом (LLM):")
        st.markdown("""
            *   **Будьте конкретны:** Задавайте открытые вопросы: *"Расскажите подробнее о вашей боли?"*, *"Когда начались симптомы?"*
            *   **Уточняйте детали:** *"Боль острая или тупая?"*, *"Есть ли еще какие-то жалобы?"*
            *   **"Проводите" физикальные обследования:** Чтобы симулировать действие, просто опишите его в чате:
                *   `Измеряю АД.`
                *   `Послушаю легкие.`
                *   `Какая у вас температура?` (можно и вопросом)
                *   `Проверю сатурацию кислорода.`
                *   `Осмотрю ваше горло.`
                *   `Пропальпирую живот.`
            LLM-пациент постарается предоставить вам клинически правдоподобные данные. Следите за изменениями! Ответы пациента могут содержать **выделенные жирным** числовые показатели.
            *   **Следите за триггерами:** Иногда пациент может сообщить новую информацию или его состояние изменится в ответ на ваши действия или по прошествии времени в диалоге.
            *   **Нужна помощь?** Используйте кнопку "Запросить консультацию у коллеги" (штраф к оценке!).
        """)
    with col2_intro:
        st.subheader("🎯 Ваши Задачи:")
        st.markdown("""
            *   **Собрать полный и релевантный анамнез**, выявив все ключевые моменты.
            *   **Сформулировать предварительный диагноз.**
            *   **Предложить адекватный план обследования и лечения.**
            *   Продемонстрировать логику клинического мышления.
            *   Уложиться в отведенное **время**, если таймер активен.
        """)
        st.subheader("⚖️ Система Оценки:")
        st.markdown("""
            После того как вы введете диагноз и план и нажмете "Завершить и Оценить" (или время истечет), LLM-преподаватель проанализирует вашу работу.
            Оценка (0-10 баллов) будет учитывать:
            *   Полноту сбора анамнеза (включая выявление ключевых моментов).
            *   Корректность предложенного диагноза.
            *   Адекватность и полноту предложенного плана.
            *   Проведение необходимых "обследований".
            *   Общую клиническую логику.
            *   Использование консультаций (может незначительно снизить балл).
            Вы получите подробную обратную связь, включая **Протокол консультации** и анализ на **Частые ошибки** для данного типа случая.
        """)
    st.markdown("---")
    st.info("👈 **Для старта выберите или сгенерируйте сценарий в панели слева!**")

if not st.session_state.scenario_selected and not st.session_state.current_scenario:
    st.stop()
