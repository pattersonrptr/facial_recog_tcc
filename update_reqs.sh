 
#!/usr/bin/env bash

#
# Usa o comando pipreqs para gerar ou atualizar o arquivo requirements.txt
#
# pipreqs só adiciona no requirements.txt os módulos necessários para o projeto.
#
# Instalar pireqs: pip install pipreqs
#

pipreqs --force .

cat requirements.txt
