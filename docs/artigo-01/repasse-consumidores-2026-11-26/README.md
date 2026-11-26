# Artigo 01 · revisão do repasse aos artigos 02 a 09

Repasse documental para revisão sobre o commit `cf8b332`. O [dossiê binário](../expansao-2026-11-25/README.md), a [reprodução histórica da VM](../../reproducao-vm/2026-11-24/README.md) e os manifestos correspondentes já existem na base.

- [Matriz por equipe](MATRIZ_DEPENDENCIAS.md) · [CSV com versões, caminhos, hashes e comandos](MATRIZ_DEPENDENCIAS.csv).
- [Verificações](VERIFICACOES.json) · [Procedência](PROVENIENCIA_REVISAO.json) · [Acervo e canais de circulação](ACERVO_CANAIS.csv).
- [Baseline de Leonam](BASELINE_LEONAM.md) · [Rodada 3, controle e alertas](INTEGRACAO_R3_E_CONTROLE.md) · [Insumos UNSW para Gabriel](UNSW_INSUMOS.md).

## Três referências experimentais

| Referência | Alvo e partição | Resultado reutilizável |
| --- | --- | --- |
| Original, FedAvg 3 rodadas | BENIGN × DDoS, 254.074 exemplos de treino, cinco clientes, seed 2305. | Inferência unitária de 81 medidas, score e embedding. SHAP com o mesmo global e cinco locais regenerados. |
| Histórico, FedAvg-SGD × SCAFFOLD, 20 rodadas | BENIGN × DDoS, ajuste 203.251, validação interna 50.815. Candidata por cliente com seed 2405, feita retrospectivamente. | Matrizes e scores dos modelos globais, limiares históricos e Macro-F1. A aprovação das fatias retrospectivas está pendente. |
| Expansão de `cf8b332` | BENIGN × ATTACK, protocolo candidato e quinta como validação. | Logística e CNN, comparação pareada na VM, resultados por seed no dossiê. Sexta reservada. |

Pedro pode calcular o Macro-F1 como média dos F1 de BENIGN e DDoS. Para o resultado global, somar as matrizes e só então calcular a métrica. Matrizes históricas real×previsto `[[40370,406],[84,9955]]` e `[[40369,407],[2538,7501]]` dão exatamente 0,9849740690317736 e 0,9003566947170022. Limiares `0.3651059865951538` e `0.21026940643787384` inclusivos. O F1_DDoS é separado do Macro-F1. A distribuição não propõe adotar FF1, CHI ou CER a partir desses números.

Os resultados da rodada 20 são exploratórios na validação interna usada para escolher os limiares. As fatias retrospectivas aguardam revisão de Pedro e Willian. A expansão BENIGN × ATTACK também tem resultados exploratórios, medidos na quinta. O teste de sexta permanece reservado e a divisão final aguarda revisão.

## Como obter os pacotes

Solicite a Willian o ZIP indicado para sua frente. Willian coordena o repasse pelo canal autorizado, separado do Git. O [inventário do acervo](ACERVO_CANAIS.csv) identifica o caminho de origem, tamanho e SHA-256 de cada ZIP. A publicação deste README documenta o repasse, e o recebimento dos arquivos deve ser confirmado separadamente.

| Destinatários | ZIP | Uso |
| --- | --- | --- |
| Pedro Nonato, Artigo 02 | `SecFedNet_Artigo02_Clientes_Rodada20_CANDIDATA.zip` | Associação retrospectiva, scores e Macro-F1 dos modelos globais da rodada 20. |
| Kaim Chavante, Artigo 05 | `secfednet-shap-fedavg3.zip` | Global, cinco modelos locais regenerados e índices da rodada 3. |
| Mateus Vasconcelos e Marcos Gabriel, Artigos 06 e 07 | `secfednet-cnn1d-v1.zip` | Pesos, contrato de inferência e embedding da rodada 3 para integração. O contrato de alertas em revisão está na [fixture sintética](alerta_sintetico.json). |

Leonam começa pela [ficha da baseline](BASELINE_LEONAM.md) e pelas partições históricas referenciadas nela. Marina e Juliany acompanham na [matriz](MATRIZ_DEPENDENCIAS.md) a dependência do detector multiclasse e das entregas dos Artigos 03 e 07. Willian e Gabriel têm os manifestos e locks referenciados no [inventário](ACERVO_CANAIS.csv) para a auditoria de cadeia. A matriz detalha o escopo disponível de cada frente.

Ao receber um ZIP, calcule `sha256sum -- NOME_DO_ZIP`, compare com o inventário, extraia em pasta nova e rode `sha256sum -c SHA256SUMS` sobre o manifesto interno. O ZIP de Pedro traz `verificar_entrega.py`, o de Kaim traz `validar_pacote.py` e o de inferência traz o comando CPU e o JSON de exemplo. Para conferência conjunta no acervo local, use os comandos abaixo.

## Tabela para o manuscrito

A revisão usa as contagens da auditoria nominal posterior, distintas da expansão binária. As 11 variantes Attempted são agregadas a partir do CSV sem incluir TOTAL. Só BENIGN tem suporte simultâneo em treino e validação. A independência de episódios de ataques do mesmo tipo segue pendente. [CSV analítico](tabela_manuscrito.csv) · [Episódios](episodios_por_classe.csv).

| Rótulo CICIDS2017 corrigido | Bruto | Attempted | Não finito | Excl. grupo | Excl. bruto | Excl. float32 | Treino | Validação | Teste reservado |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BENIGN | 1.582.566 | 0 | 5 | 547.691 | 39.310 | 41 | 536.862 | 170.113 | 288.544 |
| Botnet | 736 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 736 |
| DDoS | 95.144 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 95.144 |
| DoS GoldenEye | 7.567 | 0 | 0 | 6.311 | 0 | 0 | 1.256 | 0 | 0 |
| DoS Hulk | 158.468 | 0 | 0 | 126.499 | 0 | 0 | 31.969 | 0 | 0 |
| DoS Slowhttptest | 1.740 | 0 | 0 | 1.592 | 0 | 0 | 148 | 0 | 0 |
| DoS Slowloris | 3.859 | 0 | 0 | 3.859 | 0 | 0 | 0 | 0 | 0 |
| FTP-Patator | 3.972 | 0 | 0 | 62 | 0 | 0 | 3.910 | 0 | 0 |
| Heartbleed | 11 | 0 | 0 | 0 | 0 | 0 | 11 | 0 | 0 |
| Infiltration | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 36 | 0 |
| Infiltration - Portscan | 71.767 | 0 | 0 | 0 | 14.794 | 0 | 0 | 56.973 | 0 |
| Portscan | 159.066 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159.066 |
| SSH-Patator | 2.961 | 0 | 0 | 27 | 0 | 0 | 2.934 | 0 | 0 |
| Web Attack - Brute Force | 73 | 0 | 0 | 50 | 0 | 0 | 0 | 23 | 0 |
| Web Attack - SQL Injection | 13 | 0 | 0 | 13 | 0 | 0 | 0 | 0 | 0 |
| Web Attack - XSS | 18 | 0 | 0 | 18 | 0 | 0 | 0 | 0 | 0 |
| 11 rótulos Attempted | 11.979 | 11.979 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| TOTAL | 2.099.976 | 11.979 | 5 | 686.122 | 54.104 | 41 | 577.090 | 227.145 | 543.490 |

## Conferência

Na pasta desta seleção:

```bash
sha256sum -c SHA256SUMS
python3 -I -B gerar_tabela_readme.py
python3 -I -B verificar_alerta_sintetico.py alerta_sintetico.json
```

Da raiz do acervo, após obter a distribuição original corrigida em canal autorizado:

```bash
dist='secfednet-local/artigo-01/execucoes/2026-11-26/entrega-consumidores-11-33-26/distribuicao'
py="$PWD/secfednet-local/artigo-01/cnn-flower/.venv/bin/python"
(cd "$dist" && sha256sum -c SHA256SUMS)
python3 -I -B "$dist/verificar_pacotes.py" --python-torch "$py"
```

Os três ZIPs passaram isoladamente em pasta temporária. O manifesto da seleção cobre os arquivos efetivamente incluídos aqui. A distribuição tem seu próprio manifesto. As referências externas e os ZIPs permanecem no acervo com tamanho e SHA em [ACERVO_CANAIS.csv](ACERVO_CANAIS.csv). Alertas reais, controle de transferência, produção de SBOM e assinaturas permanecem dependentes de revisão ou produção indicada na matriz.
