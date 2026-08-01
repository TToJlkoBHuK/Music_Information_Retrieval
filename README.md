# Преобразование музыки в читаемый нотный лист

Наверняка вы все заходили на платформу YouTube и смотрели видеоролики вот такого формата:
<img width="1278" height="718" alt="image" src="https://github.com/user-attachments/assets/17e33ade-f089-4d3c-81a8-a682aca9f0ce" />

Так вот, этот формат называется Piano visualizator. Особенность заключается в том, что экран видео поделен примерно на 2 части:
1) Нижняя часть занимает фортепиано, которое может быть как и настоящим, так и чисто симуляцией:
   <img width="1794" height="429" alt="image" src="https://github.com/user-attachments/assets/b4b75351-da3f-4e42-9240-5fbaf0df3abc" />
   <img width="1795" height="409" alt="image" src="https://github.com/user-attachments/assets/5c9c7a22-668d-44ac-b0c4-f1e8cdc5b886" />
2) Верхняя часть представляет из себя симуляцию падающих блоков разной длины, интерпретируемые на человеческий язык как ноты разной длины
   <img width="1796" height="603" alt="image" src="https://github.com/user-attachments/assets/23d2732a-3095-440a-9372-87ced0fbb40f" />
   <img width="1793" height="624" alt="image" src="https://github.com/user-attachments/assets/9e083d9d-13b1-47a5-9aab-ec24f5e926c9" />

<hr>

## Основные понятия нотного листа

Нотный стан - основа для записи музыкальных нот, обязательно содержит ключи, полосу из 5 линий, тактовую черту
<img width="1187" height="707" alt="image" src="https://github.com/user-attachments/assets/23b74c46-2956-41f1-ad46-eb174c386d0c" />

Расположение нот в зависимости от нажатия на нотной клавиатуре. В классическом фортепиано - 9 октав.
<img width="1600" height="755" alt="image" src="https://github.com/user-attachments/assets/f088c785-564e-4564-9218-bb04c8ded5e7" />

Длительность нот и пауз
<img width="910" height="326" alt="image" src="https://github.com/user-attachments/assets/59f45e7b-e130-441b-aef1-55ef543dc2c6" />

Тональности - это когда нужно повысить тон той или иной ноты, в басовом ключе - это бемоль, в скрипичном ключе - диез. Отмена любого из них называется бекар
<img width="1079" height="930" alt="image" src="https://github.com/user-attachments/assets/eb0c14ef-4a8f-4194-88f4-a39baa0991a2" />

Наиболее встречаемые нотные знаки
<img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/682fb457-8b2c-4af5-bc6b-070b47336bd4" />

<hr>

## Проблематика

Проблема заключается в том, что посмотрел видеоролик, человеку явно захочется выучить произведение и сыграть его на инструменте, но найти полные ноты на бесплатных площадках или невозможно или с риском подхватить вирус. В большинстве своем - надо заплатить автору, что то аналога Boosty и Patreon. Также большинство всех роликов - для англоязычной аудитории, почти не встречались русскоязычные авторы.\
Существует ряд готовых opensource проектов, которые позволяют расшифровывать ноты, но они в основном, опять же, для англоязычной аудитории и имеют свои недостатки, такие как отсутствие отслеживания тональности, скорости, тембра и так далее. Или плохо распознают с прямой ссылки клавиатуру или нужно скачивать видео.

<hr>

## Идея

Используя компьютерное зрение, методы анализа звука, возможно еще какие-нибудь методы машинного обучения для распознавания тонкостей написания нот, предполагается устранить все недостатки opensource проектов и сделать на русской локализации в виде приложения.

## Используемый стек и фреймворки
![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
