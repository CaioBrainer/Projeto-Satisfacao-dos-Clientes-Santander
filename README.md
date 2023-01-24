<p align="center">
<img width="700" alt="f1" src="https://user-images.githubusercontent.com/92734524/214183342-4e2224b4-f1c0-46a6-ae7a-51d4e59306bd.jpg">
</p>

<h1> Projeto Satisfacao dos Clientes Santander</h1>
 

<p align="justify">
Neste projeto de aprendizado de máquina, iremos trabalhar com centenas
de recursos anônimos (~370 variáveis preditoras) para prever se um cliente 
está satisfeito ou insatisfeito com
sua experiência bancária. Definiremos o problema de negócio e realizaremos a preparação dos
dados, com o uso de alguns algoritmos treinaremos um modelo com acurácia de
pelo menos 70%.
 </p>

# Tabela de conteúdos 

1. [Definição do Problema de Negócio](https://github.com/CaioBrainer/Projeto-Satisfacao-dos-Clientes-Santander#definição-do-problema-de-negócio)
2. [Análise Exploratória dos Dados e Insights](https://github.com/CaioBrainer/Projeto-Satisfacao-dos-Clientes-Santander#analise-exploratoria-dos-dados-e-insights)
3. [Algoritmos Base e Grid Search CV](https://github.com/CaioBrainer/Projeto-Satisfacao-dos-Clientes-Santander#algoritmos-base-e-grid-search-cv)

# Definição do Problema de Negócio
<p align="justify">
A satisfação do cliente é uma medida fundamental de sucesso. Clientes
insatisfeitos cancelam seus serviços e raramente expressam sua insatisfação antes
de sair. Clientes satisfeitos, por outro lado, se tornam defensores da marca!
O Banco Santander disponibilizou um dataset com centenas
de recursos anônimos (~370 variáveis preditoras) para que nós cientistas de dados pudessemos analisar e
identificar clientes insatisfeitos no início do relacionamento. Tal analise preditiva seria benéfica
para que o Santander adotasse medidas proativas para melhorar a felicidade de um cliente antes que
seja tarde demais. 
</p>

# Análise Exploratória dos Dados e Insights 
<p align="justify">
 A análise preditiva do dataset é bastante desafiadora, pois o dataset possui uma grande quantidade de observações e de variáveis anônimas.
 Além do grande número de variáveis, observamos que o dataset é extremamente desbalanceado, contando com alguns valores outliers em algumas colunas
 e possuindo colunas com valores únicos em todas as observações, o que poderia prejudicar a performance do modelo.
</p>


<p align="center">
<img width="400" alt="f2" src="https://user-images.githubusercontent.com/92734524/214185795-6aa5a1f6-5b0e-458d-a94d-57430b2abbcf.jpeg">
</p>

<p align="center">
Histograma com as variáveis alvo do projeto, 0 está para clientes satisfeitos e 1 para clientes insatisfeitos
</p>


# Algoritmos Base e Grid Search CV
<p align="justify">
O modelo escolhido para avaliar esse dataset foi o XGBoostClassifier, uma implementação otimizada do gradiente boosting. Em primeiro momento
foi utilizado sua versão base com todos os hiperparâmetros padrões, neste primeiro momento não foi obtido um resultado muito satisfatório já que
o modelo falhava em identificar a classe correspondente aos clientes insatisfeitos.
</p>


<p align="center">
<img width="650" alt="f3" src="https://user-images.githubusercontent.com/92734524/214185415-a25965f7-add6-44c0-9021-58e86378919e.jpeg">
</p>

<p align="center">
Matriz de confusão com os resultados do modelo base, apresentando um resultado muito ruim.
</p>

<p></p>

<p align="justify">
Após realizar uma busca de hiperparâmetros com GridSearchCV (omitido do código pelo longo tempo de execução), obtivemos hiperparamêtros 
que otimizaram a classificação e fazendo um balanceamento de pesos para as classes em conjunto foi possível a identificação de 72% dos clientes insatisfeitos nos dados de teste.
O modelo também apresenta uma precisão de 95% em conjunto com uma revocação de 79%, ambas balanceadas.
</p>

<p align="center">
<img width="650" alt="f4" src="https://user-images.githubusercontent.com/92734524/214185628-f244b85e-d9fe-4a1a-b2c0-ce3e3f4979a2.jpeg">
</p>

<p align="center">
Matriz de confusão com os resultados do modelo pós tunning dos hiperparâmetros, agora apresentando bons valores de classificação.
</p>


<p align="center">
<img width="400" alt="f5" src="https://user-images.githubusercontent.com/92734524/214188876-6f8a1853-4b36-4c91-a453-7aa6ef09b81f.jpg">
</p>

<p align="center">
Classification report do modelo final, apresentando acurácia de 79%, uma precisão de 95% e revocação de 79%.
</p>

