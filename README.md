# Reduce

## Запуск и сборка
Сборка осуществляется с помощью Conan и Cmake, результат компиляции -- консольное приложение. Необходимые зависимости OpenCl и OpenMP

## Использование 

```
reduce.exe  [ --device-type { gpu | cpu | any } ]
            [ --device-index index ]
```

- `device-type` -- выбор типа устройства, на котором запустится подсчет на OpenCL
- `device-index` -- так как устройств одного типа может быть несколько, можно выбрать. При выбранном типе any сначала выбор происходит среди GPU, потом среди CPU


## Исполнение
При запуске запустится 2 теста: первый -- тест корректности, второй -- тест производительности. Данные для тестов генерируются случайно

В первом случае выведется референсная сумма, подсчитанная однопоточно. Далее выводится время подсчета на OpenMP и его результат, а потом результаты на OpenCL.

В тесте производительности выводится только время подсчета.

## Полученные результаты
```
Device: NVIDIA GeForce RTX 3050 Laptop GPU      Platform: NVIDIA CUDA
Validity test
Test array size: 107107
Reference sum 8521.03
CPU ans: 8521.2
Time in ms: 5.307
GPU ans: 8521.18
Time in ms without memory: 0.026624
Time in ms with memory: 0.06912

Large test
CPU Time in ms: 98.998
GPU Time in ms without memory: 22.7369
GPU Time in ms with memory: 323.834
```
Конфигурация ноутбука:

- CPU: 12th Gen Intel i7-12700H 45W (20) Total Cores 14, Performance-cores 6, Efficient-cores 8, Total Threads 20
- DGPU: NVIDIA GeForce RTX 3050 Mobile, 4 GB GDDR6
- Memory: 32 GB, DDR5
