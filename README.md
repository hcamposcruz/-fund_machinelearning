# Relatório de Aula Prática - Fundamentos de Machine Learning


## Fundamentos de Machine Learning

**Autor:** Hudson de Campos Cruz

---

## Script de Árvore de Decisão

Este script tem como objetivo utilizar árvores de decisão para classificar solicitações de empréstimo com base nas informações fornecidas. O script realiza a leitura de um arquivo CSV contendo dados dos clientes, processa esses dados, treina um modelo de árvore de decisão e avalia seu desempenho.

### Script

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Importação dos Dados a partir de um Arquivo CSV (separado por ;)
df = pd.read_csv('dados_clientes.csv', delimiter=';')

# 2. Carregamento e Preparação dos Dados
# Convertendo variáveis categóricas em numéricas
df['historico_credito'] = df['historico_credito'].map({'bom': 1, 'ruim': 0})
df['emprego'] = df['emprego'].map({'empregado': 1, 'desempregado': 0})
df['propriedade'] = df['propriedade'].map({'sim': 1, 'nao': 0})
df['classe'] = df['classe'].map({'conceder': 1, 'negar': 0})

# Separando recursos (X) e rótulos (y)
X = df.drop('classe', axis=1)
y = df['classe']

# 3. Divisão dos Dados em Conjunto de Treinamento e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Treinamento da Árvore de Decisão
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 5. Avaliação da Árvore de Decisão
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy * 100:.2f}%')

# 6. Classificação e Predição
def classificar_emprestimo(idade, renda, historico_credito, emprego, propriedade):
    dados_cliente = pd.DataFrame({
        'idade': [idade],
        'renda': [renda],
        'historico_credito': [1 if historico_credito == 'bom' else 0],
        'emprego': [1 if emprego == 'empregado' else 0],
        'propriedade': [1 if propriedade == 'sim' else 0]
    })
    resultado = clf.predict(dados_cliente)
    return 'conceder' if resultado[0] == 1 else 'negar'

# Exemplo de uso da função de classificação
resultado = classificar_emprestimo(30, 5000, 'bom', 'empregado', 'sim')
print(f'Resultado da classificação: {resultado}')

```

---

## Instalação das Bibliotecas
Para execução deste script foi necessário a instalação das bibliotecas pandas e scikit-learn:

### Importação das Bibliotecas

**import pandas as pd**: Importa a biblioteca pandas e a atribui ao alias pd. Pandas é utilizada para manipulação e análise de dados.

**from sklearn.model_selection import train_test_split**: Importa a função train_test_split da biblioteca scikit-learn, usada para dividir os dados em conjuntos de treinamento e teste.

**from sklearn.tree import DecisionTreeClassifier**: Importa a classe DecisionTreeClassifier da biblioteca scikit-learn, usada para criar e treinar o modelo de árvore de decisão.

**from sklearn.metrics import accuracy_score**: Importa a função accuracy_score da biblioteca scikit-learn, usada para calcular a acurácia do modelo.



### Carregamento e Preparação dos Dados

**pd.read_csv('dados_clientes.csv', delimiter=';')**: Carrega os dados a partir de um arquivo CSV delimitado por ;.

**map**: Converte as variáveis categóricas (historico_credito, emprego, propriedade, classe) para valores numéricos.

**X = df.drop('classe', axis=1)**: Cria a matriz de características (X) excluindo a coluna classe.

**y = df['classe']**: Cria o vetor de rótulos (y) contendo a coluna classe.



### Divisão dos Dados

**train_test_split**: Divide os dados em conjunto de treinamento e teste.

**test_size=0.3**: Define que 30% dos dados serão usados para teste.

**random_state=42**: Garante que a divisão dos dados seja reprodutível.



### Treinamento

**DecisionTreeClassifier**: Cria um modelo de árvore de decisão.

**random_state=42**: Garante reprodutibilidade.

**fit(X_train, y_train)**: Treina o modelo usando o conjunto de treinamento.



### Avaliação da Árvore

**predict(X_test)**: Gera previsões para o conjunto de teste.

**accuracy_score(y_test, y_pred)**: Calcula a acurácia comparando as previsões com os rótulos reais.

**print(f'Acurácia: {accuracy * 100:.2f}%')**: Exibe a acurácia em porcentagem.



### Função classificar_emprestimo

**pd.DataFrame**: Cria um DataFrame para os dados do cliente.

**clf.predict**: Usa o modelo treinado para prever se o empréstimo será concedido ou não.

**Retorno**: Retorna 'conceder' ou 'negar' com base na previsão.



### Uso da Função classificar_emprestimo

**classificar_emprestimo(30, 5000, 'bom', 'empregado', 'sim')**: Exemplo de chamada da função com dados específicos de um cliente.

**print(f'Resultado da classificação: {resultado}')**: Exibe o resultado da classificação.


## Conecte-se comigo no LinkedIn

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hudson-campos-cruz-005056172/)