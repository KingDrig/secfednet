# Integração da rodada 3 · Artigos 05, 06 e 07

## Modelo e dados

Arquitetura de referência em `cnn1d.py` do ZIP de inferência, e em `secfednet-local/artigo-01/cnn-flower/cnn1d.py` na baseline Git. A entrada `(N,81)` é expandida para um canal. Duas convoluções 1-D, 8 canais com kernel 5 e stride 3, depois 12 com kernel 3 e stride 2, ambas com ReLU. Average pooling adaptativo em quatro posições, flatten 48 e Linear 48→16 com ReLU. O embedding são as 16 ativações após esse ReLU (`embedding.1`), antes da cabeça Linear 16→1. O score é a sigmoide do logit de DDoS, sem calibração. Classes 0=BENIGN e 1=DDoS, limiar inclusivo 0,5. Os pesos e o mapa de classes da expansão binária pertencem a outro ensaio.

Modelo global `checkpoint_cnn1d.pt`, SHA-256 `37db7d4f6d0123786cc4ed5227d670edb1344c07089e8ff6910c6446e57400f3`. O contrato ordena as 81 medidas brutas de fluxo e o checkpoint contém média/desvio populacional do treino. Ajuste em float64, escala 1 se desvio zero, conversão em float32 para a rede. O formato de saída JSON local tem `classe_id`, `classe`, `score` e `embedding`. O verificador reproduziu o exemplo binário exato.

O CICIDS2017 corrigido usado tem ZIP SHA `97fdb91d339e2d8cf5627f981b831e5e7e400b981c58181c451a38fd03c48883`, membro `friday.csv` SHA `ebd499e6f23bd59f9cb81bec28178491b02b925fa5640a24215c9437d79482d0`. Ambos recalculados localmente. [Distribuição dos autores](https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip).

Cinco clientes simulados. Dirichlet **por classe**, α=0,5, `default_rng(2305)`, multinomial e permutação por classe. Ordena as posições do treino final para cada cliente. Repete se um cliente fica vazio, com máximo 100 tentativas na mesma RNG. A primeira foi aceita. Rodada 3 após três rodadas FedAvg, uma época local, lote 512, Adam lr 0,001, seed de pesos iniciais 23 e minibatches `23 + 100*rodada + cliente`. Partição e etapa de normalização usam 254.074 exemplos finais. O pacote SHAP tem posições do treino final base zero, ordinais lógicos base um do CSV e cinco pesos após treino local, imediatamente antes da agregação da terceira rodada. São regenerados, com identidade do global demonstrada. O cliente 2 tem 7.634 BENIGN e zero DDoS. Os hashes individuais e vínculos ficam no manifesto interno e no inventário desta revisão.

Kaim pode rodar o verificador do pacote em CPU sem dataset. Para materializar o background SHAP, use exclusivamente o conjunto de treino alinhado a `indices_particao.npz`, conforme `INDICES.md` do ZIP. O trabalho de explicação ainda requer decidir background, alvo (logit ou score) e validar o explainer. A comparação que depender de exemplos do Artigo 04 segue condicionada a esse artigo.

## Controle de rótulos embaralhados · Artigo 06

O passo 20 do roteiro pede um detector treinado com rótulos de treino embaralhados para comparação. Nenhum checkpoint desse controle foi localizado. Para produzir um controle comparável será preciso aprovar o alvo do pré-treino e a regra de embaralhamento, fixar os mesmos dados, índices de cliente, normalizador, pesos iniciais, cronograma, otimizador, orçamento e sementes, e fixar uma seed separada para a permutação. Registrar hash das labels originais e embaralhadas, preservação das contagens e justificativa para permutação global ou dentro do cliente. A validação conserva seus rótulos verdadeiros. O resultado precisa de identificação e manifesto próprios, incluindo qual checkpoint é original e qual é controle. O benchmark e a normalização de fraude devem ser verificados pela equipe de Mateus e Marcos, especialmente a dimensão de entrada e a compatibilidade do espaço de features. Esse controle também é distinto do estado de controles matemáticos do SCAFFOLD.

## Proposta de alerta · Artigo 07

O pacote atual produz inferência unitária binária. Para validar o formato do consumidor, use [o schema de proposta](alerta.schema.json) e [a fixture sintética](alerta_sintetico.json). Eles deixam explícitos ID de fluxo, tempo UTC, origem/destino, portas, protocolo, taxonomia e semântica do score. A fixture tem IPs de documentação, modelo fictício e limiar nulo. Confira com:

```bash
python3 -I -B verificar_alerta_sintetico.py alerta_sintetico.json
```

A versão proposta exige `sintetico: true`. Antes de usar em produção, Mateus e Marcos precisam revisar versão, identificação e minimização dos metadados do fluxo, contrato de transporte/privacidade, taxonomia, origem dos pesos, limiar operacional, calibração e associação com explicações. Alertas multiclasse UNSW dependem dos dados e da respectiva avaliação. O Artigo 08 depende do corpus e artefatos autorizados do Artigo 07. Validação feita pela biblioteca padrão sobre o exemplo, com verificações semânticas de campos e IP. A adoção e os alertas reais seguem pendentes.
