# Проверка статистических гипотез на больших наборах данных о биржевой активности.
### Курсовая работа
**1. Тема : Проверка статистических гипотез на больших наборах данных о биржевой активности.**\
**2. Проверить выдвинутые гипотезы по набору данных и сделать выводы и прогнозы**\
**3. Задачи:**\
**3.1 Провести предварительный анализ датасета**\
**3.3 Обработать данные**\
**3.4 Визуализировать данные и найти закономерности**\
**3.5 Выдвинуть гипотезы и проверить стат.значимость**\
**4. Ожидается выявление зависимости между разными индексами бирж а также ожидается найти закономерности изменения индексов с течением времени.**\
**5. В датасете представлены ежедневные данные о значении индексов 11 бирж в долларах США.
Представлено открытие и закрытие торгов, минимальное и максимальное значение за каждый день а также объем торгов.**
```Python
import pandas as pd
import numpy as np
import math
##############################
from scipy.stats import *
from sympy.stats import *
##############################
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateFormatter, AutoDateLocator, date2num
import matplotlib
import seaborn as sns
import folium
```
```Python
import os
path = "C:/Users/Lenovo/Birzha/data"
dir_list = os.listdir(path)
dir_list = dir_list[1:12]
```
### Выполним загрузку данных в один большой датасет, для удобства работы
#### Для наглядности уберем ненужные символы в названии индексов, которые возникли при парсинге.
```Python
main_dataframe = pd.DataFrame(pd.read_csv(dir_list[0]))
```
```Python
main_dataframe = pd.DataFrame(pd.read_csv(dir_list[0]))
for i in dir_list[1:]:
    data = pd.read_csv(i)
    df = pd.DataFrame(data)
    main_dataframe = pd.concat([main_dataframe,df],ignore_index = True)
main_dataframe['ticker'] = main_dataframe['ticker'].str.replace(r'^', '')
```
```Python
### Первый взгляд на данные:

main_dataframe
```Python
main_dataframe.shape == main_dataframe[main_dataframe['raw_close']== main_dataframe['close']].shape
```
### Уберем избыточный признак raw_close, т.к он равен признаку close на всем датасете.
```Python
main_dataframe = main_dataframe.drop(['raw_close'],axis = 1)
```
### Опишем что означает каждый признак
```Python
features = pd.DataFrame(np.array(['Биржа','Дата','Цена закрытия','Наиболее высокая цена',
                      'Наименее низкая цена','Цена открытия','Объем торгов']),main_dataframe.columns)
features
```
name = np.unique(main_dataframe['ticker'].values)
name

### Описание аббревиатуры каждого индекса бирж:
```Python
name_stock = np.array(['Шанхайская фондовая биржа','Австралийская фондовая биржа',
              'Индекс Доу Джонсона(Американская фондовая биржа)','Французская фондовая биржа',
              'Немецкая фондовая биржа','Канадская фондовая биржа',
              'Аргентинская фондовая биржа','Японская фондовая биржа',
              'Нью-Йорская фондовая биржа','(Индекс волатильности)Чикагская фондовая биржа','Американская фондовая биржа'])

abbreviation = pd.DataFrame({'Индекс': name, 'Расшифровка': name_stock})
abbreviation

main_dataframe['date'] = pd.to_datetime(main_dataframe['date'])

main_dataframe.info
```
### Описательная статистика датасета по цене открытия:
```Python
main_dataframe.groupby(['ticker'])['open'].describe()
```
### Описательная статистика датасета по цене закрытия:
```Python
main_dataframe.groupby(['ticker'])['close'].describe()
```
### Описательная статистика датасета по объему торгов:
```Python
main_dataframe.groupby(['ticker'])['volume'].describe()
```
### Посмотрим на выбросы в данных
```Python
def feature_boxplot(feature):
    fig, ax = plt.subplots(figsize = (25,18), nrows = 3, ncols = 4)
    row, col,n = 0,0,4
    for i in name:
        sns.boxplot(x = main_dataframe[main_dataframe['ticker']==i][feature],
                             color = 'blue',
                             ax = ax[row,col])
        ax[row, col].set_title(i, fontsize = 14)
        col = col + 1
        if col == n:
            col = 0
            row = row + 1

feature_boxplot('volume')
```
### Можем заметить значительные выбросы во всех биржах по объему торгов, также сразу можем понять что на бирже VIX объемы отсутствуют, что соответствует здравому смыслу, так как этот индекс показывает общую волатильность фондовогго рынка и является своего рода индикатором поведения рынка.
```Python
fig, ax = plt.subplots(figsize=(14, 9)) 
xtick_locator =  matplotlib.dates.MonthLocator()
x2_locator = matplotlib.dates.YearLocator(2, month=1, day=1)
ax.plot(main_dataframe[main_dataframe['ticker']=='000001.SS']['date'].values,
         main_dataframe[main_dataframe['ticker']=='000001.SS']['close'].values)
ax.plot(main_dataframe[main_dataframe['ticker']=='AXJO']['date'].values,
         main_dataframe[main_dataframe['ticker']=='AXJO']['close'].values)
ax.plot(main_dataframe[main_dataframe['ticker']=='DJI']['date'].values,
         main_dataframe[main_dataframe['ticker']=='DJI']['close'].values)
ax.plot(main_dataframe[main_dataframe['ticker']=='FCHI']['date'].values,
         main_dataframe[main_dataframe['ticker']=='FCHI']['close'].values)
ax.plot(main_dataframe[main_dataframe['ticker']=='GDAXI']['date'].values,
         main_dataframe[main_dataframe['ticker']=='GDAXI']['close'].values)
ax.plot(main_dataframe[main_dataframe['ticker']=='GSPTSE']['date'].values,
         main_dataframe[main_dataframe['ticker']=='GSPTSE']['close'].values)
ax.plot(main_dataframe[main_dataframe['ticker']=='MERV']['date'].values,
         main_dataframe[main_dataframe['ticker']=='MERV']['close'].values)
ax.plot(main_dataframe[main_dataframe['ticker']=='N225']['date'].values,
         main_dataframe[main_dataframe['ticker']=='N225']['close'].values)
ax.plot(main_dataframe[main_dataframe['ticker']=='NYA']['date'].values,
         main_dataframe[main_dataframe['ticker']=='NYA']['close'].values)
ax.plot(main_dataframe[main_dataframe['ticker']=='VIX']['date'].values,
         main_dataframe[main_dataframe['ticker']=='VIX']['close'].values)
ax.plot(main_dataframe[main_dataframe['ticker']=='XAX']['date'].values,
         main_dataframe[main_dataframe['ticker']=='XAX']['close'].values)
ax.xaxis.set_minor_locator(xtick_locator)
ax.xaxis.set_major_locator(x2_locator)
ax.legend(list(name))
```
![Image1.](https://github.com/KaRToXa9/Stock-Hypothesis/blob/main/pictures/tickers.png)
### Посмотрев на графики цен открытия/закрытия мы можем увидеть взаимосвязь между индексами разных бирж, а также резкие скачки и падения в определенные периоды времени.

### Построим распределения цен открытия и закрытия:

name
```Python
main_dataframe[main_dataframe['ticker']=='000001.SS']['open'].values

def feature_dens(feature):
    fig, ax = plt.subplots(figsize = (26,18), nrows = 3, ncols = 4)
    row, col,n = 0,0,4
    for i in name:
        sns.histplot(main_dataframe[main_dataframe['ticker']==i][feature].values,
                             kde = True,
                             color = 'blue',
                             ax = ax[row,col])
        ax[row, col].set_title(i, fontsize = 13)
        col = col + 1
        if col == n:
            col = 0
            row = row + 1
```

### Цена открытия:
```Python
feature_dens('open')
```
![Image1.](/pictures/distributions.png)
### Цена закрытия:
```Python
feature_dens('close')
```
![Image1.](/pictures/dist_close.png)
### Взглянув на распределение признаков индексов бирж, можно сделать однозначный вывод о том что данные распределены не нормально, и четко распознать распределение каждого признака индекса бирж трудно. Стоит это учесть при дальнейшей проверке статистических гипотез, так как параметрические методы будут неприменимы к таким данным.

### Выведем матрицу коэффициентов корреляции между признаками внутри каждой биржи:
```Python
main_dataframe
```
```Python
fig, ax = plt.subplots(figsize = (30,20), nrows = 3, ncols = 4)
row, col,n = 0,0,4
for i in name:
    sns.heatmap(main_dataframe[main_dataframe['ticker']==i].iloc[:,2:].corr(), 
                             annot=True,
                             ax = ax[row,col])
    ax[row, col].set_title(i, fontsize = 14)
    col = col + 1
    if col == n:
        col = 0
        row = row + 1
```
![Image1.](/pictures/distributions.png)
### Можем видеть что все признаки имеют сильную положительную корреляцию друг с другом, кроме признака volume. В некоторых индексах он имеет слабую отрицательную корреляцию.

### После подробного анализа и обработки данных, можно выдвигать гипотезы. Будут проверяться следующие статистические гипотезы:
**1. Гипотеза о влиянии объема торгов на цену закрытия биржи** \
**2. Равенство медианы цен открытия и закрытия биржи.**\
**3. Гипотеза о влиянии цены закрытия бирж друг на друга.**

### 1. Выдвинем и проверим гипотезу о влиянии объема торгов на цену закрытия биржи. 
**Отберем дни когда был маленький объем торгов и большой и сравним среднее в этих двух группах.\
Проверим статистически значимую разницу средней цены в дни с высоким объемом торгов и низким объемом торгов. Наша альтернативная гипотеза будет односторонняя и будет говорить нам о том что в дни с меньшим объемом торгов средняя цена закрытия была меньше чем в дни с большим объемом торгов.**

### $H_0: \mu_1 = \mu_2$
### $H_1: \mu_1 < \mu_2$

### Для проверки гипотезы будем использовать Mann-Whitney U-тест, для сравнения средних значений между двумя группами. 
**Выбор теста обоснован не нормальным распределением признака close.**

### Уберем индекс VIX и MERV из теста, так как в первом индексе отсутствуют торги, он отражает волатильность рынка, а индекс MERV в большей степени состоит из нулевых торгов, что тоже позволяет нам убрать из рассмотрения данный индекс для проведения гипотез.
**Также выберем ненулевые торги во всех оставшихся индексах, будем считать дни с большим объемом торгов где все значения объемов лежат ниже медианного значения(выбор данной величины обоснован устойчивостью медианы к выбросам, а также ненормальностью распределения признака volume).**
```Python
alpha = 0.05
con = []
p_list = []
h_list = []
l_list = []
df = main_dataframe[main_dataframe['volume'] > 0]
for stock in name[(name!='VIX') & (name!='MERV')]:
    high_volume = df[(df['ticker']==stock)
                             & (df['volume'] > df[df['ticker']==stock].volume.median())]
    h_list.append(high_volume[high_volume['ticker']==stock]['close'].mean())
    low_volume = df[(df['ticker']==stock) 
                            & (df['volume'] <= df[df['ticker']==stock].volume.median())]
    l_list.append(low_volume[low_volume['ticker']==stock]['close'].mean())
    statistic, p_value = mannwhitneyu(low_volume['close'], high_volume['close'], alternative="less")
    p_list.append(p_value)
    if p_value < alpha:
        con.append('H0 отвергается')
    else:
        con.append('H0 не отвергается')

Hyp1 = pd.DataFrame({'Биржа': name[(name!='VIX') & (name!='MERV')],'p_value':p_list,
              'Вывод':con, 'Среднее в больших объемах':h_list, 'Среднее в маленьких объемах':l_list })

Hyp1
```
### Вывод:
### При проверке гипотезы были обнаружены статистически значимые различия в биржах: 000001.SS, DJI, GSPTSE, NYA. В данных биржах можно сделать вывод о том что в дни когда объем торгов больше, цена закрытия будет больше в сравнении с днями с меньшим объемом торгов.

### Во всех остальных биржах не удалось обнаружить статистически значимых различий при уровне значимости 0,05. С вероятностью 95% можно делать вывод о том, что в дни с большим и маленьким объемом торгов, средние цены закрытия биржи не имеют статистически значимых различий между собой.

### 2. Гипотеза о равенстве медианного значения цен открытия и цен закрытия бирж.
**Выдвинем гипотезу о равенстве медианного значения цен открытия и закрытия бирж. Для оценки медианного значения по данным которые не подчиняются параметрическим методам воспользуемся бутсреп тестом.**

**Бутстрэп в статистике — практический компьютерный метод исследования распределения статистик вероятностных распределений, основанный на многократной генерации выборок методом Монте-Карло на базе имеющейся выборки. Позволяет просто и быстро оценивать самые разные статистики (доверительные интервалы, дисперсию, корреляцию и так далее) для сложных моделей.**

**Идея реализации:**\
**1. Расчитываем фактическую статистику как разницу медианных значений индекса открытия и закрытия**\
**2. Генерируем подвыборку из выборки индексов открытия и закрытия многократно с повторениями, тем самым мы моделируем поведение нашей статистики(медианного значения).**\
**3. Далее расчитываем разницу между каждым значением в двух выборках(цена открытия-цена закрытия) и рассчитываем на полученной выборке медиану.**\
**4. Строим доверительный интервал заданной вероятности и смотрим попадает ли фактическое значение нашей статистики в него или нет, на основании чего делаем вывод.**

### $H_0: Me_{open} = Me_{close}$
### $H_1: Me_{open} \neq Me_{close}$
### Уровень значимости $\alpha = 0.05$
```Python
def get_percentile_ci(bootstrap_stats, pe, alpha):
    """Строит перцентильный доверительный интервал."""
    left, right = np.quantile(bootstrap_stats, [alpha / 2, 1 - alpha / 2])
    return left, right
```
### Пример на бирже FCHI, как работает бутстреп-тест:
```Python
B = 10000
alpha = 0.05
```
```Python
opening_index = main_dataframe[main_dataframe['ticker'] == 'FCHI'].open.values
closing_index = main_dataframe[main_dataframe['ticker'] == 'FCHI'].close.values

pe = np.median(closing_index) - np.median(opening_index)

bootstrap_values_a = np.random.choice(opening_index, (B, len(opening_index)), True)
bootstrap_metrics_a = np.median(bootstrap_values_a, axis=1)
bootstrap_values_b = np.random.choice(closing_index, (B, len(closing_index)), True)
bootstrap_metrics_b = np.median(bootstrap_values_b, axis=1)
bootstrap_stats = bootstrap_metrics_b - bootstrap_metrics_a
ci = get_percentile_ci(bootstrap_stats, pe, alpha)
has_effect = not (ci[0] < 0 < ci[1])

print(f'Фактическое значение статистики: {pe:0.2f}')
print(f'{((1 - alpha) * 100)}% доверительный интервал: ({ci[0]:0.2f}, {ci[1]:0.2f})')
print(f'Отличия статистически значимые: {has_effect}')
```
Фактическое значение статистики: -3.01 \
95.0% доверительный интервал: (-39.05, 37.69) \
Отличия статистически значимые: False 
### Запустим бутстреп тест на всех биржах и выведем результат проверки гипотез.

**В данной функции аргументами являются уровень значимости alpha и количество извлеченных подвыборок**
```Python
def bootstrap_stock_mean(alpha, B = 2000):
    eff_list =[]
    pe_list = []
    for stock in name:
        opening_index = main_dataframe[main_dataframe['ticker'] == stock].open.values
        closing_index = main_dataframe[main_dataframe['ticker'] == stock].close.values

        pe = np.median(closing_index) - np.median(opening_index)
        bootstrap_values_a = np.random.choice(opening_index, (B, len(opening_index)), True)
        bootstrap_metrics_a = np.median(bootstrap_values_a, axis=1)
        bootstrap_values_b = np.random.choice(closing_index, (B, len(closing_index)), True)
        bootstrap_metrics_b = np.median(bootstrap_values_b, axis=1)
        bootstrap_stats = bootstrap_metrics_b - bootstrap_metrics_a
        ci = get_percentile_ci(bootstrap_stats, pe, alpha)
        has_effect = not (ci[0] < 0 < ci[1])
        if has_effect:
            has_effect = 'H0 отвергается'
        else:
            has_effect = 'H0 не отвергается'
        eff_list.append(has_effect)
        pe_list.append(pe)
    df = pd.DataFrame({'Биржа' : name, 'Значимость' : eff_list, 'Фактическое значение статистики': pe_list})
    return df
```
```Python
Hyp2 = bootstrap_stock_mean(0.05)
```
```Python
Hyp2
```
### Визуализируем наше поведение статистик в биржах для наглядности:
```Python
def plot_bootstrap_mean(name):
    fig, ax = plt.subplots(figsize = (27,20,), nrows = 3, ncols = 4)
    row, col,n = 0,0,4
    for stock in name:
        opening_index = main_dataframe[main_dataframe['ticker'] == stock].open.values
        closing_index = main_dataframe[main_dataframe['ticker'] == stock].close.values

        pe = np.median(closing_index) - np.median(opening_index)
        bootstrap_values_a = np.random.choice(opening_index, (B, len(opening_index)), True)
        bootstrap_metrics_a = np.median(bootstrap_values_a, axis=1)
        bootstrap_values_b = np.random.choice(closing_index, (B, len(closing_index)), True)
        bootstrap_metrics_b = np.median(bootstrap_values_b, axis=1)
        bootstrap_stats = bootstrap_metrics_b - bootstrap_metrics_a  
        ci = get_percentile_ci(bootstrap_stats, pe, alpha)
        
        ax[row, col].hist(bootstrap_stats, bins=30, edgecolor='k')
        ax[row, col].axvline(x=pe, color='r', linestyle='dashed', linewidth=2, label='Фактическая статистика')
        ax[row, col].set_title(stock, fontsize = 14)
        ax[row, col].plot(ci, [-11*2, -11*2], label='перцентильный ДИ', color = 'g', linewidth=4)
        ax[row, col].legend()
        col = col + 1
        if col == n:
            col = 0
            row = row + 1
```
![Image1.](/pictures/bootstrap.png)
```Python
plot_bootstrap_mean(name)
```
### Таким образом по результатам проверки данной гипотезы мы обнаружили что во всех рассматриваемых биржах медианное значение цены открытия совпадает с ценой закрытия с доверительной вероятностью в 95% Следовательно это позволяет нам не отвергать гипотезу о равенстве медианных значений открытия и закрытия цен. В рамках бирж это может указывать на то что за сутки цена изменяется незначительно в рамках предыдущего дня и можно утверждать с 95 % вероятностью что он останется на прежнем уровне.

### 3. Выдвинем гипотезу о влиянии индексов друг на друга, будем предполагать, что изменение одного индекса влияет на изменение другого. Для этого построим матрицу корреляций цен закрытия бирж(в качестве признака возьмем цену закрытия).
```Python
result = pd.DataFrame({'date': main_dataframe[main_dataframe.ticker =='AXJO']['date'].values,
             'AXJO': main_dataframe[main_dataframe.ticker =='AXJO']['close'].values})
for i in name[name!='AXJO']:
    result = pd.merge(result,pd.DataFrame({'date': main_dataframe[main_dataframe.ticker ==i]['date'].values,
             i: main_dataframe[main_dataframe.ticker ==i]['close'].values}), how="inner", on='date')

result
```
```Python
plt.figure(figsize=(9,7))
sns.heatmap(result.corr(), annot = True, cmap = 'plasma');
```
![Image1.](/pictures/cor_matrix2.png)
### По корреляционной матрице можем сразу заметить, что индес VIX(отражающий общую волатильность) имеет слабую  отрицательную корреляцию со всеми остальными индексами бирж. Также можем отметить что индекс Шанхайской фондовой биржи имеет умеренную силу связи с остальными индексами, что на общей картине корреляционной матрицы позволяет нам сделать предварительный вывод о том что данных индекс имеет слабое влияние на остальной рынок, или же рынок слабо влияет на данный индекс.

### Проведем уже проверенный бутсреп тест, чтобы проверить гипотезу о влиянии индексов бирж друг на друга. Будем считать что индексы бирж сильно влияют друг на друга если r(коэффициент корреляции пирсона) принимает значения больше 0,75.

### $H_0: r > 0.75$
### $H_1: r< 0.75$
### Уровень значимости $\alpha = 0.05$

### Идея реализации такая же что и для подсчета медианного значения, только мы попарно сравниваем два индекса и выводим для них корреляцию для которой расчитан интервал.
```Python
def bootstrap_hypothesis_test(data, alpha, B=2000):
    tickers = name.tolist()
    pairs = []
    correlation_intervals = []

    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            ticker1 = tickers[i]
            ticker2 = tickers[j]
            closing_index1 = data[ticker1].values
            closing_index2 = data[ticker2].values

            observed_corr, _ = pearsonr(closing_index1, closing_index2)

            bootstrap_correlations = []
            for _ in range(B):
                indices = np.random.randint(0, len(closing_index1), len(closing_index1))
                bootstrap_corr, _ = pearsonr(closing_index1[indices], closing_index2[indices])
                bootstrap_correlations.append(bootstrap_corr)

            lower = np.percentile(bootstrap_correlations, (alpha/2) * 100)
            upper = np.percentile(bootstrap_correlations, (1 - alpha/2) * 100)
            interval = (lower, upper)

            pairs.append((ticker1, ticker2))
            correlation_intervals.append(interval)

    result_df = pd.DataFrame({'Пара бирж': pairs, 'Корреляционный интервал': correlation_intervals})
    result_df['Попадение в интервал'] = result_df['Корреляционный интервал'].apply(lambda x: abs(x[1]) > 0.75)

    return result_df

corr_stock  = bootstrap_hypothesis_test(result,0.05)
corr_stock
```
### Выведем список пар бирж для которых крайнее значение 0.75 попало в интервал.
```Python
true_corr = corr_stock[corr_stock['Попадение в интервал']==True]
true_corr

true_corr.head(12)
```
### Так как получилось большое количество пар индексов где корреляция выше 0.75, визуализацию проводить не будем, лишь удостоверимся на одной паре бирж, что мы корректно провели бутсреп тест.
```Python
closing_index1 = result['AXJO'].values
closing_index2 = result['XAX'].values
pairs = []
correlation_intervals = []

observed_corr, _ = pearsonr(closing_index1, closing_index2)

bootstrap_correlations = []
for _ in range(2000):
    indices = np.random.randint(0, len(closing_index1), len(closing_index1))
    bootstrap_corr, _ = pearsonr(closing_index1[indices], closing_index2[indices])
    bootstrap_correlations.append(bootstrap_corr)

    lower = np.percentile(bootstrap_correlations, (alpha/2) * 100)
    upper = np.percentile(bootstrap_correlations, (1 - alpha/2) * 100)
    interval = (lower, upper)

    correlation_intervals.append(interval)

plt.hist(bootstrap_correlations, bins=30, edgecolor='k', alpha=0.5)
plt.axvline(observed_corr, color='red', linestyle='dashed', linewidth=2, label='Observed Statistic')
plt.xlabel('Bootstrap Statistics')
plt.plot(interval, [-11*2, -11*2], label='перцентильный ДИ', color = 'g', linewidth=4)
plt.ylabel('Frequency')
plt.title('AXJO ~ XAX')
plt.legend()
plt.show()
```
![image.png](/pictures/example.png)
### После проведения гипотезы, нам удалось найти биржи , цены закрытия которых сильно связаны между собой. Список приведен ниже в виде таблицы
```Python
true_corr
```
### Построим граф взаимодействия различных бирж, для этого на карте укажем связи между городами где располагаются биржи.
```Python
abbreviation['city'] = ['Shanghai','Sydney','New York','Paris','Berlin','Ottawa','Buenos Aires','Tokyo',
                         'New York','New York','New York']
```
### Воспользуемся датасетом с геолокацией городов мира и отберем нужные нам города где располагаются биржжи
```Python
location = pd.read_csv('C:/Users/Lenovo/Курсовая/Локация/worldcities.csv')
```
```Python
merged = pd.merge(abbreviation,location, on ='city', how='inner')
```
### Зададим координаты городов
```Python
cord = [(31.2165, 121.4365),
 (-33.92, 151.1852),
 (40.6943, -73.9249),
 (40.6943, -74.2249),
 (40.6943, -73.6249),
 (40.6943, -73.1249),
 (48.8667, 2.3333),
 (52.5218, 13.4015),
 (45.4167, -75.7),
 (-34.6025, -58.3975),
 (35.685, 139.7514)]
```
### Воспользуемся результатом объединения попарного сравнения бирж и их координат.
```Python
df4 = pd.read_csv('Попарное сравнение.csv')
df4 = df4.drop('Unnamed: 0', axis=1)
```
### Настраиваем отображение карты, добавим выделение страны, а также постиоим граф связей бирж
```Python
from ast import literal_eval

x = [literal_eval(df4.cord_x.values[i]) for i in range(len(df4.cord_x.values))]

y = [literal_eval(df4.cord_y.values[i]) for i in range(len(df4.cord_y.values))]

pair_c = list(zip(x, y))

import json
with open('C:/Users/Lenovo/Birzha/Локация/countries.geojson') as handle:
    country_geo = json.loads(handle.read())

Country = ['Japan','Argentina','France','Australia','Canada','Germany','United States of America']

country = []
for i in country_geo['features']:
    if i['properties']['ADMIN'] in Country:
        country.append(i)

import folium

m = folium.Map(tiles = 'openstreetmap')
for i in range(len(cord)):
    folium.Marker(
    cord[i], tooltip=merged['Индекс'].unique()[i]
).add_to(m)
    
for pair in pair_c:
    folium.PolyLine(pair).add_to(m)
    
for i in country:
    folium.GeoJson(i,
               name=i['properties']['ADMIN']).add_to(m)

folium.TileLayer('stamenterrain').add_to(m)
```
### Полученный граф связей на карте мира:

![image.png](/pictures/map.png)
Hyp1

Hyp2
```Python
con = pd.merge(Hyp1.iloc[:,[0,2]],Hyp2.iloc[:,[0,1]], on='Биржа', how='right')

con = con.rename(columns={'Вывод':'Гипотеза об объеме торгов','Значимость':'Гипотеза о равенстве медианы'})

con['Гипотеза о взаимодействии бирж'] = [0,8,7,6,8,8,6,7,8,0,4]
```
### Выводы:

### Ниже приведена таблица с гипотезами по порядку и их выводами, в третьей гипотезе указано количество бирж с которыми взаимодействует конкретная биржа.
```Python
con
```
![image.png](/pictures/изображение_2023-11-20_210355743.png)
### Проделав данную работу, мы смогли провести анализ и предобработку датасета с дневными данными об индексах мировых фондовых бирж. Распределения признаков индексов бирж тяжело отнести к конкретному распределению и оценить параметрическими методами. 

### Для проверки гипотез на больших выборках, с ненормально и непараметрически распределенными величинами нужно пользоваться непараметрическими тестами и критериями, в частности в проделанной работе был выбран бутстреп-тест, который позволил нам смоделировать поведение наших проверяемых статистик.
###  В ходе работы проверили три гипотезы:
***Гипотеза о влиянии объема торгов на бирже*** позволила нам сделать вывод о том что не было обнаружено различий в средних ценах при больших и маленьких объемах торгов, что может послужить отличным инструментом для принятия решений на тех биржах, где было выялвено влияние объема торгов на цену закрытия, и позволит отследить объемы сделок за конкретный день и сделать прогноз повысится цена или понизится в конце дня. \
***Гипотеза о равенстве медианных значений цен закрытия и открытия*** подтвердилась практически для всех бирж.\
***Гипотеза о влиянии цен закрытия бирж друг на друга*** также позволила нам взглянуть на биржи которые имеют сильную зависимость друг от друга. Данная гипотеза помогает выявить взаимосвязи между биржами находящимися на разных концах земли. Это поможет принимать взвешанные решения о торгах на фондовых рынках,  а также учесть как поведет себя конкретная биржа, если пройдут изменения на зависящих от нее других бирж.

### Проверка статистических гипотез на наборе данных биржевой активности позволяет делать выводы и строить прогнозы. Однако нужно учитывать что на любой признак в биржах влияют множество факторов которые тяжело предугадать. С этим связана трудность выбора критериев и тестов проверки, отличным вариантом станут непараметрические критерии, или реализация тестов для моделирования параметров которые необходимо оценить. По результатам проведенных гипотез, можно выделить взаимодействия между биржами, также поведение цен бирж зависящее от объемов торгов и поведение цен открытия и закрытия в рамках одного дня. Найденные закономерности можно использовать как инструмент для прогнозирования и принятия решений на рынке бирж.
