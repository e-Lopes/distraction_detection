# Ferramentas auxiliares

- `docker/g48a-insightface.Dockerfile`: ambiente isolado para InsightFace G48A.
  O contexto de build continua sendo a raiz do repositório.
- `facial-landmarks-benchmark/`: benchmark independente, com README, Compose,
  código e ambientes próprios. Execute o Makefile/Compose a partir dessa pasta.

Estas ferramentas não são a entrada principal do pipeline. Para os experimentos
atuais, execute `python -m fase_2 interface` na raiz do repositório.
