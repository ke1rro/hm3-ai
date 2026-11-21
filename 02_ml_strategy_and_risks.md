# HW3 — ML Strategy, Risks & Things to Be Aware Of

> **Статус документа:** інженерні рекомендації та дослідницькі гіпотези для AI-generated text detection. **Не є вимогами організаторів, готовим звітом чи доведеними властивостями конкретного датасету.** Усі твердження про очікувану користь підходів потрібно перевіряти локальними експериментами.
>
> Офіційні умови — у [`01_assignment_requirements.md`](./01_assignment_requirements.md).

## 1. Основна ціль

Побудувати **надійний multiclass text classifier** (`human`, `ai`, `fusion`), що максимізує **private-leaderboard Macro-F1**, але при цьому є відтворюваним, пояснюваним та не використовує data leakage.

Ключові інженерні принципи:

1. **Validation first:** довіряти локальному експерименту більше, ніж коливанням public LB.
2. **Fixed reference:** змінювати одну гіпотезу за раз, порівнюючи з чітко визначеною моделлю.
3. **Per-class diagnostics:** `fusion` важливий так само, як `human` та `ai`.
4. **Multilingual/domain robustness:** перевіряти якість за мовами, доменами та довжинами.
5. **Budget awareness:** обмежений GPU-час і дедлайн — аргумент на користь спочатку простих і надійних моделей.

## 2. Ризик №1 — Leakage через споріднені тексти

За умовою, human-текст і його AI-/fusion-версії не перетинаються між **офіційними** train/test. Але якщо самостійно зробити random train/validation split, споріднені версії **можуть** опинитися по різні боки.

**Наслідок:** модель розпізнає спільний контент або topic, а не реальні сигнали авторства; local Macro-F1 стає надмірно оптимістичним.

### Що перевірити

- Exact duplicates після обережної нормалізації.
- Near-duplicates / сильно схожі документи (наприклад char n-gram similarity, MinHash; лише на `train.csv`).
- Чи існує поле/закономірність, що дозволяє відновити групи версій одного джерела, не використовуючи зовнішні дані.
- Частоту близьких пар між класами, особливо `human` ↔ `fusion` та `human` ↔ `ai`.

### Що робити

- За наявності надійних груп — **group-aware train/validation split** із контролем класів/мов/доменів наскільки це можливо.
- Якщо групи не відновлюються надійно — fixed stratified split плюс leakage audit, із явним зазначенням залишкового ризику.
- **Не** підганяти групування під validation score та не використовувати test для пошуку/навчання груп.

## 3. Ризик №2 — Domain / language shift

Текстові ознаки можуть відрізнятися за мовою та доменом. Науковий текст і блог мають різні стилі незалежно від авторства.

**Гіпотеза:** модель може вивчити `domain`/`lang` як shortcut, якщо співвідношення класів нерівномірне. Це треба **перевірити**, а не вважати фактом.

### Обов'язкові зрізи для аналізу

- `label × lang`: counts, normalized shares.
- `label × domain`: counts, normalized shares.
- `label × lang × domain`: за наявності достатньої кількості прикладів.
- Кожен зріз: support, macro-F1, F1 для класів, confusion matrix (де вистачає спостережень).
- За потреби: stability across buckets за кількістю слів або токенів.

**Метадані `lang` і `domain` є і в train, і в test.** Їх можна розглядати як features або для окремої діагностики, але потрібно валідовувати, чи вони справді допомагають поза власними груповими кореляціями.

## 4. Ризик №3 — Long texts / truncation

Довжини: **20–3000 слів** (медіана 463); типові encoder backbones часто мають обмеження, наприклад **512 токенів**. Водночас кількість **слів ≠ кількість токенів**, а tokenization може відрізнятися за мовою.

Якщо модель бачить лише перші токени, вона може пропустити AI-rewritten фрагменти ближче до кінця документа. Особливо це важливо перевірити для `fusion`.

### Варіанти експериментів

1. **Head truncation** — перші `max_length` токенів (контрольний варіант).
2. **Head + tail** — початок і кінець документа в межах того самого token budget.
3. **Sliding windows / chunks** — кілька фрагментів із перекриттям, агреговані до document prediction.
4. **Hierarchical/MIL aggregator** — навчувана агрегація multiple chunk embeddings/logits на рівні документа, якщо вистачить compute і часу.

Порівнювати **на тих самих документах, тим самим backbone та validation split**. Окрім Macro-F1 вимірювати runtime, VRAM, inference latency.

## 5. Особливість `fusion`: локальна зміна стилю

`fusion` — частково LLM-переписаний людський текст; це не означає, що його embedding завжди лежить «посередині» між `human` та `ai`.

### Гіпотеза

Локальні відмінності між chunks можуть бути кориснішими за ознаки всього документа.

Умовний приклад (не реальна розмітка):

```text
Document: [human] [human] [AI-rewritten] [AI-rewritten] [human]
Label:    fusion
```

### Важливе обмеження supervision

У датасеті наведено **document-level labels**. Не можна автоматично стверджувати, що кожен chunk документа `fusion` теж має label `fusion` або знати, які саме chunks переписані.

Можливі document-level aggregations:

- mean logits / mean probabilities між chunks;
- max / top-k pooling для локальних сигналів (ризик false positives);
- statistics of chunk probabilities (mean, max, variance) у маленькому meta-classifier;
- Multiple Instance Learning / learned attention pooling.

Обережно: навчати meta-classifier **на тих самих in-sample predictions**, які отримані при навчанні базових моделей, не можна без належного out-of-fold або окремого validation protocol — це leakage/overfitting.

## 6. Базовий ML pipeline

```text
train.csv
   |
   v
Data audit / EDA / duplicate check
   |
   v
Fixed leakage-aware train/validation split
   |
   +-----------------------------+
   |                             |
   v                             v
TF-IDF word/char            Multilingual encoder
+ Linear models             (XLM-R / mDeBERTa)
   |                             |
   +-------------+---------------+
                 |
                 v
       Per-class error analysis
                 |
                 v
       Extra experiments
       (long text, stylometry,
        class weights, ensemble)
                 |
                 v
      Select model on validation
                 |
                 v
      Train final configuration
      using allowed train data
                 |
                 v
     Test inference + ID alignment
                 |
                 v
      submission.csv
```

## 7. Моделі та експерименти: пріоритети

| Підхід | Гіпотеза / мета | Compute | Пріоритет |
|---|---|---|---|
| Word TF-IDF + Logistic Regression | Зрозумілий fast baseline | CPU | **P0** |
| Char TF-IDF + Logistic Regression / Linear SVM | Стиль, пунктуація, морфологічні шаблони | CPU | **P0** |
| Multilingual XLM-R-base fine-tuning | Contextual representations | GPU | **P0** |
| Head+tail vs head-only | Точніший сигнал із довгих текстів | GPU inference/training | **P1** |
| Chunk aggregation | Потенційний виграш для `fusion` | GPU inference/training | **P1** |
| Stylometric features | Додаткові сигнали щодо авторського стилю | CPU | P2 |
| Class weights / logit bias tuning | Баланс per-class F1 за Macro-F1 | CPU/невеликий GPU | P2 |
| Probability blending | Компенсація різних типів помилок | CPU | P2 |
| Other backbone | Архітектурна диверсифікація | GPU | P3 |
| SHAP / LIME / token attribution | Explainability bonus, висновки про сигнали | Variable | P3 |

**P0** — стартовий мінімум; **P1** — головні додаткові експерименти; **P2/P3** — після отримання надійного основного результату.

## 8. Які feature engineering ideas протестувати

**Classical text features:**

- Word n-grams, char n-grams і їх комбінація.
- TF-IDF `min_df`, `max_features`, `ngram_range`, regularization `C`.
- Проста модель на embeddings **не заміняє** required fine-tuned multilingual encoder.

**Stylometry** (як дозволений додатковий експеримент):

- Довжина тексту, кількість речень, середня/дисперсія довжин речень.
- Середня довжина слова, lexical diversity (з поправкою на довжину).
- Частки знаків пунктуації, цифр, великих літер.
- Частка повторів, типи переходів, лексична однорідність між chunks.

**Застереження:** не всі ознаки однаково осмислені англійською/німецькою; обчислення ознак може залежати від токенізатора. Усі trainable transformations — `fit` тільки на train portion.

**Perplexity-based features:** можливі лише з дозволених open-weight pretrained models і без зовнішніх даних/закритих API; враховувати різні токенізатори та мовну залежність. Це **додаткова гіпотеза**, а не гарантований сигнал AI-авторства.

## 9. Коректна схема експериментів

Для кожного run записувати:

```text
experiment_id
hypothesis
train_split_id / validation_split_id
training_subset_id + sample seed (if applicable)
model / checkpoint / tokenizer
preprocessing / max_length / chunking
hyperparameters / train epochs
macro_f1 + per_class_precision_recall_f1
metrics by lang/domain/length
runtime / GPU memory (when relevant)
confusion_matrix artifact
observations + failure cases
```

Рекомендована логіка експерименту:

1. Сформулювати **одну** чітку гіпотезу.
2. Визначити **fixed reference model**.
3. Зафіксувати train/validation split і training subset.
4. Змінити тільки цільовий компонент (або явно обґрунтувати інші зміни).
5. Порівняти Macro-F1, F1 для `fusion`, помилки за мовою/доменом/довжиною.
6. Задокументувати негативний результат так само, як позитивний.

**Не слід:** виконувати багато tuning iterations на одному маленькому validation і вважати його неупередженою оцінкою. Після вибору конфігурації корисна додаткова robustness-перевірка (коли дозволяє час), без використання public LB як validation.

## 10. Ансамблювання без overfitting

Кандидат на перевірку — probability blending:

\[
P(y\mid x)=\alpha P_{\text{encoder}}(y\mid x)
+(1-\alpha)P_{\text{TF-IDF}}(y\mid x).
\]

- Перед blending перевірити **однаковий порядок класів** у prediction arrays.
- Порівняти окремі моделі на фіксованому split і подивитися, чи їхні помилки **доповнюють** одна одну.
- Не перебирати сотні ваг на невеликому validation наборі — це теж overfitting.
- `LinearSVC` не видає `predict_proba` без калібрування; калібрування вимагає коректного CV/holdout всередині train.
- Якщо будується meta-model, використовувати **out-of-fold** predictions або належну holdout-схему.
- Після вибору ваг/конфігурації можна refit базові моделі на всьому дозволеному train, якщо процедура відтворювана.

## 11. Macro-F1: чому не оптимізуємо тільки accuracy

Оцінка:

\[
F1_{\rm macro}=\frac{F1_{human}+F1_{ai}+F1_{fusion}}{3}.
\]

Покращення більшості класів не гарантує зростання Macro-F1, якщо суттєво погіршується `fusion`.

Тому дивитися:

- Confusion matrix у count і row-normalized вигляді.
- `fusion → human`, `fusion → ai` і зворотні помилки.
- Precision/recall/F1 кожного класу та їхню підтримку (`support`).
- Динаміку метрики на різних довжинах, особливо довгих `fusion` текстах.

**Thresholds / logit bias:** для multiclass можна досліджувати зміщення class logits або інші правила вибору класу, але лише на validation із контролем перебору; стандартний `argmax` — обов'язковий reference.

## 12. Public vs private leaderboard

Public score отримується на приблизно 30% test, private — на решті 70%. Різниця між ними можлива **без** явного overfitting: наприклад через sample variability. Але частий підбір моделей за public score створює додатковий ризик leaderboard overfitting.

Тому:

- Використовувати public LB як **secondary sanity check** після локальної оцінки.
- Не оптимізувати ensemble weights за public submissions.
- Обирати final submissions на основі стабільності local validation та змістовної відмінності двох кандидатів.
- Не робити висновок «модель generalizes» лише з одного public score.

## 13. План за часом (дата підготовки: 21 листопада 2026)

| Коли | Робота | Deliverable |
|---|---|---|
| **21.11** | Data/duplicate audit, EDA, fixed split, TF-IDF baselines, перший submission | Надійний local Macro-F1 + працюючий CSV |
| **22.11** | XLM-R fine-tuning, порівняння з baseline, long-text/chunk експеримент | Encoder результат + щонайменше два змістовні експерименти (за можливості) |
| **23.11** | Error analysis, невеликий blending/class-bias search, robustness, final inference | До **2 final Kaggle submissions** до 23:59 Kyiv |
| **24.11** | Відтворюваність, cleanup коду, **самостійне** написання технічного звіту | Код + звіт в LMS до 23:59 Kyiv |

Не варто запускати додаткове дороге навчання, якщо це загрожує вчасному коректному submission.

## 14. Що досліджувати на першому EDA

```text
1. Schema, nulls, ids, label balance
2. Label balance by language
3. Label balance by domain
4. Label × language × domain intersections
5. Word / char / token lengths (different languages and labels)
6. Exact duplicates and near-duplicate candidates within train
7. Representative text examples (human, ai, fusion)
8. Class-vs-length relationships / unusually short and long texts
9. Potential formatting artifacts (headers, whitespace, markup)
10. Tokenization truncation rate at chosen max_length
```

Для кожного спостереження відповідати: **«Яке модельне рішення або validation-ризик із цього випливає?»**

Наприклад:

| Якщо EDA виявило… | Гіпотеза / наступний крок |
|---|---|
| Багато довгих текстів, які обрізаються | Порівняти head-only vs head+tail / chunks. |
| Сильний дисбаланс `fusion` | Моніторити F1 і recall `fusion`; протестувати class weights або logit bias. |
| Помітні near-duplicates | Перевірити, чи потрібне group-aware split. |
| Значна різниця помилок між `en-EN`/`de-DE` | Оцінити multilingual backbone/feature robustness. |
| Різні типи помилок у TF-IDF і encoder | Перевірити невеликий probability blend. |

## 15. Ризики дослідження та контрольні заходи

| Ризик | Наслідок | Mitigation |
|---|---|---|
| Leakage між training та validation | Завищена оцінка | Group-aware split / duplicate audit, train-only fitting |
| Відкидання частини довгих текстів | Втрата сигналу, потенційно `fusion` | Long-text experiment, truncation audit |
| Надмірна прив'язка до домену/мови | Слабкий transfer | Stratified diagnostics, conditional metrics |
| Переоптимізація public LB | Падіння private score | Fixed local validation, мало submissions |
| Fusion chunk label noise | Некоректне supervision | Документні labels, MIL/aggregation, без псевдо-розмітки chunks як факту |
| Обмежений GPU budget | Не встигаємо final model | CPU baseline, fixed 20–50k subset, early valid submission |
| Over-tuning ensemble | Випадковий local gain | Невеликий заздалегідь визначений search і robustness check |
| Невідтворювані результати | Втрата балів на захисті | Seeds, версії, run config, чистий CLI/notebooks |
| Порушення правил | Штраф/анулювання | Лише дозволені дані, pretrained weights; AI tools не пишуть report |

## 16. Definition of Done

- [ ] Є надійний fixed validation protocol і leakage audit.
- [ ] Є сильний класичний baseline та fine-tuned multilingual encoder.
- [ ] Є **мінімум два** додаткових зафіксованих експерименти з порівнянням проти reference.
- [ ] Відомі Macro-F1, per-class metrics, confusion matrix та зрізи за мовою/доменом/довжиною.
- [ ] Обрано фінальну модель за локальною валідацією, а не випадковим public LB gain.
- [ ] Final inference вирівняно за `id`, CSV перевірено.
- [ ] Усі рішення можна пояснити на one-on-one defense.
- [ ] Технічний звіт студент пише самостійно згідно з правилами.

---

**Ключова думка:** конкурентний результат тут може залежати не лише від backbone, а від **якості розбиття**, **довгих документів**, **здатності відрізняти `fusion`** та **поміркованого ансамблю без leakage і leaderboard overfitting**.
