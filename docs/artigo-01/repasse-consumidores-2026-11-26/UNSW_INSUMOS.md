# UNSW-NB15 · insumos a recuperar por Gabriel

- Corpus integral. `UNSW-NB15_1.csv`, `UNSW-NB15_2.csv`, `UNSW-NB15_3.csv`, `UNSW-NB15_4.csv`.
- Dicionário. `UNSW-NB15_features.csv`, identificado na ficha de Gabriel pela variante `NUSW-NB15_features.csv`.
- Ground truth e lista de eventos. `UNSW-NB15_GT.csv` e `UNSW-NB15_LIST_EVENTS.csv`.
- Partições oficiais, se disponíveis. `UNSW_NB15_training-set.csv` e `UNSW_NB15_testing-set.csv`.
- Alternativa local para recuperar os arquivos de referência. `Arquivo compactado.zip`, SHA-256 documentado `04a48a9e386a8a3836d9d807d2ad1502c3a6e68c34788a71d534c3b644470b82`.

As partições oficiais esperadas têm SHA `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa` (treino) e `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559` (teste). As contagens disponíveis, 175.341 e 82.332, são documentais. O inventário do corpus integral por classe depende da leitura dos quatro CSVs e dos metadados. [Fonte institucional UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset) · [Tabela documental](unsw_cobertura_documental.csv).

Na recuperação, registrar procedência, data, nomes, tamanho e SHA de cada arquivo. Confirmar esquema, chaves e contagens em nova pasta. Para os dois CSVs oficiais, usar `docs/artigo-01/expansao-2026-11-25/gabriel/unsw-nb15/count_classes.py --source CAMINHO --out-dir NOVO_DESTINO` via Python 3. Auditar duplicatas, rótulos conflitantes e interseção de grupos/vetores antes de combinar versões ou treinar. A eventual validação das partições oficiais sai somente de seu treino. O Artigo 07 precisa, depois, de um detector multiclasse avaliado e metadados autorizados de fluxo antes de gerar alertas reais.
