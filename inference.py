import os
import torch
from unsloth import FastLanguageModel

# Пути берем из окружения
LORA_PATH = os.getenv("LORA_PATH", "./lora_qwen_telegram")
MODEL_PATH = os.getenv("MODEL_PATH", "./my_model") 

print("="*50)
print(f"ЗАГРУЗКА МОДЕЛИ И LORA АДАПТЕРОВ ИЗ: {LORA_PATH}")
print("="*50)

# Unsloth корректно подтянет базовую модель и наложит адаптеры
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = LORA_PATH,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Включаем оптимизацию Unsloth
FastLanguageModel.for_inference(model)

# ================= НАСТРОЙКА ТЕСТОВЫХ ДАННЫХ =================

sys_prompt_city = "Ты — редактор городского Telegram-канала Москвы. Твоя задача: писать короткие и понятные новости города. Обязательно начинай пост с одного тематического эмодзи. Пиши в нейтрально-информативном, но живом стиле. Текст должен состоять из 1-2 коротких абзацев. Если есть источник фото или видео, указывай его отдельной строкой в конце."

sys_prompt_scandal = "Ты — редактор скандального Telegram-канала. Твоя задача: писать короткие, агрессивные и кликбейтные посты на основе скучных фактов. Используй гиперболизацию, начинай с шокирующего утверждения и ссылайся на источники для придания веса."

# 5 вопросов для Городского канала
questions_city = [
    "Напиши новость о том, что мэрия Москвы внедряет цифровые двойники (BIM-модели) для мониторинга ремонта дорог. Пилотный проект стартует на МКАД.",
    "В промзоне 'Руднево' открыли новый вычислительный кластер для тестирования промышленных нейросетей, закупив серверы с H100.",
    "Житель юго-запада Москвы во дворе дома полностью перебрал двигатель внутреннего сгорания старых Жигулей. Соседи жалуются на запах бензина и шум.",
    "Аналитики зафиксировали рост числа вакансий на Senior AI Engineer в промышленном секторе Москвы на 25% после весенней волны сокращений.",
    "На Калужском шоссе водитель потерял управление из-за заклинившей дроссельной заслонки, машина вылетела в кювет. Видео предоставил: msk_dtp."
]

# 5 вопросов для Скандального канала
questions_scandal = [
    "IT-компании заставляют разработчиков перерабатывать ради победы в гонке RAG-моделей. Многие увольняются и ищут спокойную работу со стандартным 8-часовым графиком.",
    "Суд отказался принимать иск, потому что юристы составили его с использованием корпоративного сленга, назвав финансового директора 'фиником', а адвоката 'лоером'.",
    "Роскомнадзор заблокировал очередной протокол VLESS, пользователи массово жалуются на падение кастомных прокси-серверов.",
    "Студента отчислили на первом курсе за неуспеваемость, а через несколько лет он стал ведущим инженером и теперь зарабатывает больше своих бывших преподавателей.",
    "Компания закупила кластер видеокарт на миллионы рублей, но инженеры случайно сломали конфигурацию при попытке развернуть мультиагентную систему на LangGraph."
]

# ================= ФУНКЦИЯ ДЛЯ ПРОГОНА И ЗАПИСИ =================

def run_evaluation(system_prompt, questions, output_filename):
    print(f"\nНачинаем генерацию для файла: {output_filename}")
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"СИСТЕМНЫЙ ПРОМПТ:\n{system_prompt}\n")
        f.write("="*70 + "\n\n")
        
        for i, user_query in enumerate(questions, 1):
            print(f"Обработка вопроса {i}/{len(questions)}...")
            f.write(f"ВОПРОС {i}:\n{user_query}\n\n")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
            
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize = True,
                add_generation_prompt = True,
                return_tensors = "pt",
            ).to("cuda")

            # --- ГЕНЕРАЦИЯ БЕЗ LORA ---
            with model.disable_adapter():
                outputs_base = model.generate(
                    input_ids = inputs,
                    max_new_tokens = 200,
                    use_cache = True,
                    temperature = 0.7
                )
                text_base = tokenizer.batch_decode(outputs_base[:, inputs.shape[1]:], skip_special_tokens = True)[0]
                f.write(f"❌ ОТВЕТ БАЗОВОЙ МОДЕЛИ (Без LoRA):\n{text_base.strip()}\n\n")

            # --- ГЕНЕРАЦИЯ С LORA ---
            outputs_lora = model.generate(
                input_ids = inputs,
                max_new_tokens = 200,
                use_cache = True,
                temperature = 0.7 
            )
            text_lora = tokenizer.batch_decode(outputs_lora[:, inputs.shape[1]:], skip_special_tokens = True)[0]
            f.write(f"✅ ОТВЕТ ДООБУЧЕННОЙ МОДЕЛИ (С LoRA):\n{text_lora.strip()}\n\n")
            f.write("-" * 70 + "\n\n")

# Запуск тестов
run_evaluation(sys_prompt_city, questions_city, "outputs_type1.txt")
run_evaluation(sys_prompt_scandal, questions_scandal, "outputs_type2.txt")

print("\n✅ Сравнение завершено! Результаты сохранены в 'outputs_type1.txt' и 'outputs_type2.txt'.")