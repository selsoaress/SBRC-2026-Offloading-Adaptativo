# Offloading Adaptativo para EEDNNs com Agentes de Multi-Armed Bandits (Kit de Experimentação para o SBRC 2026)

Kit de reprodução de experimentos para o artigo sobre equilíbrio e compartilhamento de recursos através de Multi-Armed Bandits.

## Estrutura do Projeto

* app.py: Código fonte da aplicação e simulação.
* requirements.txt: Lista de dependências do projeto.
* * repositório `notebooks`: contendo jupyter notebook de construção dos experimentos.

## Instalação

Recomenda-se o uso de um ambiente virtual para isolar as dependências.

1. Clone o repositório ou baixe os arquivos para uma pasta local.

2. Crie e ative o ambiente virtual:

    Windows:
    python -m venv venv
    .\venv\Scripts\activate

    Linux / Mac:
    python3 -m venv venv
    source venv/bin/activate

3. Instale as bibliotecas necessárias:

    pip install -r requirements.txt

## Execução

Com o ambiente virtual ativo, inicie a interface de simulação:

    streamlit run app.py
