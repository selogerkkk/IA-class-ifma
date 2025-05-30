#!/bin/bash

# Ativar ambiente virtual se existir
if [ -d "wine_env" ]; then
    echo "Ativando ambiente virtual..."
    source wine_env/bin/activate
fi

# Executar o projeto
echo "Executando análise de vinhos..."
python wine_analysis_simple.py 