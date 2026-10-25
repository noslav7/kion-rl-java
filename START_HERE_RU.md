# Начни здесь — Java-версия второго тестового

**Всё решение написано на Java.** Для быстрого первого запуска Maven не обязателен: в архиве уже есть готовый JAR (Java 21 bytecode).

1. Распакуй ZIP проекта, открой PowerShell в папке `kion-rl-java`.
2. Проверь Java: `java -version` (21+).
3. Выполни:

   ```powershell
   java -Xmx3g -jar kion-rl-java-1.0.0.jar
   ```

   Датасет KION скачается из официального GitHub автоматически, если доступ к GitHub разрешён.

4. Если скачивание не удалось, скачай **`interactions.csv.zip`** вручную отсюда: https://github.com/irsafilo/KION_DATASET и выполни:

   ```powershell
   java -Xmx3g -jar kion-rl-java-1.0.0.jar --data "C:\Users\<ваш пользователь>\Downloads\interactions.csv.zip"
   ```

5. После успешного запуска пришли файлы:

   - `outputs/kion/metrics.csv`
   - `outputs/kion/EXPERIMENT_REPORT.md`
   - `outputs/kion/run_config.txt`

**Не отправляй результаты `outputs/demo/` как научные — там только искусственные данные.**

## Требование к версии Java

Используй **JDK 21 LTS** (например, Eclipse Temurin 21). Проверь `java -version` и `javac -version` — в обоих случаях должна быть версия 21. В Maven зафиксировано `<maven.compiler.release>21</maven.compiler.release>`.

## Если нужна сборка из Java-исходников

```powershell
mvn -q -DskipTests package
java -Xmx3g -jar target/kion-rl-java-1.0.0.jar
```

## Если реальный запуск идёт слишком медленно

Быстрый профиль уже включён по умолчанию. Для ещё более короткого диагностического запуска на **настоящих данных**:

```powershell
java -Xmx3g -jar kion-rl-java-1.0.0.jar --max-train 300 --sft-epochs 1 --rl-steps 60 --max-eval 70
```

Однако предпочтительнее основной профиль, потому что чрезмерное сокращение обучения ослабляет исследование.

## Ограничения, которые нужно честно сказать при защите

- Это уменьшенная GPT-2-подобная Transformer-архитектура, а не исходная GPT-2 от OpenAI.
- Не используются Python/PyTorch/Colab: обучение и RL реализованы на Java.
- RL может увеличить *предсказанное* вознаграждение, не улучшив реальную удовлетворённость пользователей.
- Метрики на ограниченном train-only каталоге не являются официальным benchmark KION.
- Реальные результаты будут получены только после запуска на официальном архиве KION.

Детали и формулы описаны в `README.md`.
