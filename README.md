# Offloading Adaptativo para EEDNNs com Agentes de Multi-Armed Bandits  
## Kit de Experimentação — SBRC 2026

Este repositório contém o kit de reprodução de experimentos do artigo que investiga mecanismos de equilíbrio e compartilhamento de recursos em sistemas de offloading adaptativo para Early-Exit Deep Neural Networks (EEDNNs), utilizando Multi-Armed Bandits (MABs).

## Estrutura do Projeto

- `app.py`: Código-fonte principal da aplicação e da simulação.
- `requirements.txt`: Lista de dependências do projeto.
- `notebooks/`: Diretório de Jupyter Notebooks utilizados para a construção e análise dos experimentos.

## Instalação

Recomenda-se o uso de um ambiente virtual para isolar as dependências do projeto.

1. Clone o repositório ou baixe os arquivos para um diretório local.

2. Crie e ative um ambiente virtual:

   Windows:
   python -m venv venv  
   .\venv\Scripts\activate  

   Linux / macOS:
   python3 -m venv venv  
   source venv/bin/activate  

3. Instale as dependências necessárias:

   pip install -r requirements.txt

## Execução

Com o ambiente virtual ativado, execute a interface de simulação baseada em Streamlit:

streamlit run app.py

A aplicação será iniciada localmente e permitirá a visualização e interação com os cenários de offloading e aprendizado por reforço estudados no artigo.
