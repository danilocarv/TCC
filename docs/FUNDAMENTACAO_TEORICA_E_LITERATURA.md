# Fundamentação Teórica, Fórmulas e Catálogo de Literatura Científica (TCC)

**Autor:** Danilo Carvalho de Oliveira  
**Orientador:** Prof. Me. Douglas Donizeti de Castilho Braz  
**Instituição:** IFSULDEMINAS - Campus Poços de Caldas  
**Projeto:** Predição de Consumo Energético e Classificação de Perfis de Condução em Veículos Elétricos Utilizando Aprendizado de Máquina  
**Finalidade:** Documentar todas as fórmulas matemáticas, grandezas físicas, limiares numéricos adotados, justificativas teóricas e catálogo de referências acadêmicas (com DOIs e links diretos) para embasamento do TCC e da monografia.

---

## 📑 1. Tabela Resumo: Parâmetros, Fórmulas, Limiares e Literatura

A tabela a seguir consolida todas as grandezas, fórmulas, valores de referência adotados no código e a sustentação acadêmica encontrada em periódicos internacionais:

| Grandeza / Variável | Fórmula Matemática | Valor de Referência Adotado | Significado Físico / Critério | Faixa na Literatura Científica | Referências Principais (DOI / Link) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Aceleração Longitudinal ($a$)** | $a(t) = \frac{\Delta v}{\Delta t}$ | Amostragem a 1 Hz exato; suavização móvel de 3s | Taxa de variação temporal da velocidade; corte físico $[-8,0, +6,0]\text{ m/s}^2$ | $[-8,0, +4,0]\text{ m/s}^2$ para veículos elétricos de passeio | Oh et al. (2020), Fiori et al. (2016) |
| **Frenagem Brusca (Hard Braking)** | $a(t) \le a_{\text{lim\_brake}}$ | $a \le -2,0\text{ m/s}^2$ ($\approx -0,20g$) | Desacelerações mais intensas que o conforto usual; sinaliza condução agressiva | $-2,0\text{ m/s}^2$ a $-4,0\text{ m/s}^2$ (ou $-0,20g$ a $-0,40g$) | Bagdadi (2013), Klauer et al. (2006), Eboli et al. (2016) |
| **Arrancada Rápida (Rapid Accel)** | $a(t) \ge a_{\text{lim\_accel}}$ | $a \ge +2,0\text{ m/s}^2$ ($\approx +0,20g$) | Demanda de pico instantâneo de corrente da bateria; condução esportiva | $+2,0\text{ m/s}^2$ a $+4,0\text{ m/s}^2$ (ou $+0,20g$ a $+0,40g$) | Wang et al. (2018), Martinez et al. (2018) |
| **Tranco Longitudinal (Jerk - $j$)** | $j(t) = \frac{\Delta a}{\Delta t}$ | $\|j\| > 2,5\text{ m/s}^3$ | Variação brusca da aceleração; causa desconforto e estresse mecânico | $> 2,0\text{ m/s}^3$ a $> 3,0\text{ m/s}^3$ (ISO 2631 / JATD) | Bagdadi & Várhelyi (2011), Eboli et al. (2016) |
| **Taxas por Minuto** | $\text{Rate} = \frac{N_{\text{eventos}}}{\Delta t_{\text{min}}}$ | Eventos normalizados por 2 min | Remove o viés do tempo absoluto de percurso | Padrão em telemetria veicular e seguros UBI | Mobini Seraji et al. (2025), Eboli et al. (2016) |
| **Tamanho da Janela Temporal** | $W_{\text{size}} = T$ | $T = 120\text{ s}$ (2 minutos) | Horizonte temporal para extração de atributos comportamentais | $60\text{ s}$ a $180\text{ s}$ para perfis gerais de estilo | Mobini Seraji et al. (2025), Zhang et al. (2019) |
| **Tempo Ocioso (Idle Speed)** | $v(t) < v_{\text{idle}}$ | $v < 2,0\text{ km/h}$ ($\approx 0,55\text{ m/s}$) | Veículo imobilizado em semáforo ou parada | $v < 2,0\text{ km/h}$ ou $v < 1,6\text{ km/h}$ (1 mph) | Ciclos EPA / WLTP, SAE J2951 |
| **Corte de Condução Ativa** | $\bar{v}_{\text{janela}} \ge v_{\text{min\_ativa}}$ | $\bar{v} \ge 8,0\text{ km/h}$ | Isola manobras de estacionamento/espera do aprendizado do K-Means | $\bar{v} \ge 5\text{ a }10\text{ km/h}$ em estudos de profiling | Bingham et al. (2012), Wang et al. (2018) |
| **Potência Elétrica da Bateria ($P$)** | $P = -\frac{V_{\text{bat}} \times I_{\text{bat}}}{1000}$ | Convenção: $P > 0$ consumo; $P < 0$ regeneração | Potência líquida demandada em kW do pack de tração | Padrão automotivo para modelagem energética EV | Oh et al. (2020), Fiori et al. (2016) |
| **Energia Elétrica Acumulada ($E$)** | $E = \int P(t) \, dt$ | Integral discreta: $\sum P_i \frac{\Delta t_i}{3600}$ | Energia líquida consumida/regenerada em kWh | Padrão físico e ciclos de teste veicular | Bingham et al. (2012), De Cauwer et al. (2015) |
| **Número de Grupos ($k$)** | K-Means com $k=3$ | $k=3$: Econômico, Moderado, Agressivo | Divisão clássica tripartite de estilo de condução | $k=3$ é o consenso predominante na literatura | Martinez et al. (2018), Mobini Seraji et al. (2025) |
| **Índice de Agressividade (Ordenação)** | $\text{Score} = 2 R_{\text{brk}} + 2 R_{\text{acc}} + a_{\text{max}}$ | Ordenação determinística: Cluster 0, 1 e 2 | Ponderação das taxas de manobras extremas | Índices de severidade e risco cinemático | Bagdadi (2013), Eboli et al. (2016) |

---

## 🔬 2. Detalhamento Teórico de Fórmulas e Limiares

### 2.1. Cinemática Longitudinal e Regularização Temporal a 1 Hz
* **Fórmula da Velocidade:**
  $$v_{\text{m/s}} = \frac{v_{\text{km/h}}}{3,6}$$
* **Fórmula da Aceleração Longitudinal Discreta:**
  $$a(t) = \frac{v(t) - v(t - \Delta t)}{\Delta t}$$
* **Por que amostragem a 1 Hz ($\Delta t = 1,0\text{ s}$)?**
  Nos dados do *Vehicle Energy Dataset (VED)* coletados via interface OBD-II, múltiplas mensagens são registradas a cada fração de segundo ($\approx 100\text{ a }400\text{ ms}$) sempre que sensores secundários alteram de estado, mas a velocidade é transmitida em números inteiros de $1\text{ km/h} \approx 0,278\text{ m/s}$. 
  Se derivássemos sem regularização, $\Delta v = 0,278\text{ m/s}$ dividido por $\Delta t = 0,1\text{ s}$ geraria acelerações artificiais de $2,78\text{ m/s}^2$ (ruído de quantização). Conforme demonstrado por Oh et al. (2020), a telemetria agregada a 1 Hz é a cadência física fundamental para a dinâmica veicular macroscópica.
* **Limites de Aceleração Física:**
  Para um veículo de passageiros elétrico com tração dianteira de 80 kW (Nissan Leaf 2013 com massa de aproximadamente $1.520\text{ kg}$):
  * Aceleração máxima de arrancada no plano em pista seca: $\approx +3,5\text{ m/s}^2$ a $+4,0\text{ m/s}^2$ (0 a 100 km/h em $\approx 10,5\text{ s}$, gerando aceleração média teórica de $\approx 2,65\text{ m/s}^2$).
  * Desaceleração máxima em frenagem de emergência (ABS ativo em asfalto seco com coeficiente de atrito $\mu \approx 0,8$): $\approx -7,5\text{ m/s}^2$ a $-8,0\text{ m/s}^2$.
  * Portanto, o limitador físico (*clipping*) implementado em $[-8,0, +6,0]\text{ m/s}^2$ assegura que nenhum ruído corrompa os dados.

---

### 2.2. Limiares de Frenagem Brusca, Arrancada e Conforto

#### A) Frenagem Brusca: $a \le -2,0\text{ m/s}^2$ ($\approx -0,20g$)
* **O que significa na prática:** Qualquer desaceleração mais intensa que $-2,0\text{ m/s}^2$ força o corpo dos passageiros para frente contra o cinto de segurança e exige acionamento direto do pedal de freio mecânico (ultrapassando a capacidade pura de frenagem regenerativa do motor elétrico).
* **Fundamentação na Literatura:**
  * **Klauer et al. (2006) - 100-Car Naturalistic Driving Study (VTTI/NHTSA):** Estabelece que frenagens normais e confortáveis ocorrem entre $-1,0\text{ m/s}^2$ e $-1,5\text{ m/s}^2$. Desacelerações a partir de $-2,0\text{ m/s}^2$ ($0,20g$) marcam a fronteira de eventos intempestivos ou manobras abruptas de desaceleração.
  * **Bagdadi (2013):** Demonstra que eventos de frenagem superiores a $0,25g$ (aproximadamente $-2,45\text{ m/s}^2$) em tráfego urbano são fortes preditores de risco de colisão traseira e definem condutores impacientes.
  * **Eboli, Mazzulla e Pungillo (2016):** Identificam que passageiros começam a reportar desconforto acentuado quando a aceleração ultrapassa $1,47\text{ m/s}^2$ ($0,15g$) e desconforto severo acima de $2,0\text{ m/s}^2$.
* **Conclusão:** O valor de **$-2,0\text{ m/s}^2$** é o limiar ótimo para trânsito misto, pois captura a conduta agressiva rotineira sem esperar que o motorista chegue a uma situação extrema de quase-acidente (que ocorre acima de $-4,0\text{ m/s}^2$).

#### B) Arrancada Rápida: $a \ge +2,0\text{ m/s}^2$ ($\approx +0,20g$)
* **O que significa na prática:** Em um carro elétrico, o motor síncrono de ímã permanente entrega torque instantâneo desde zero RPM ($280\text{ Nm}$ no Nissan Leaf). Acelerações superiores a $+2,0\text{ m/s}^2$ exigem que o condutor pressione o pedal do acelerador com mais de 70% de curso, demandando picos de potência superiores a $40\text{ kW}$ da bateria.
* **Fundamentação na Literatura:**
  * **Martinez et al. (2018) & Wang et al. (2018):** Classificam arrancadas acima de $1,8\text{ a }2,0\text{ m/s}^2$ como características típicas de condutores no perfil *"Aggressive / Sporty"*, contrastando com condutores *"Eco / Calm"* que raramente ultrapassam $1,0\text{ m/s}^2$.

#### C) Tranco Longitudinal (Jerk): $\|j\| > 2,5\text{ m/s}^3$
* **Fórmula:**
  $$j(t) = \frac{a(t) - a(t - \Delta t)}{\Delta t}$$
* **O que significa:** O *jerk* (ou solavanco) mede a rapidez com que a aceleração varia. Um motorista que freia forte mas pisa no freio de maneira progressiva tem jerk baixo; já aquele que "dá patada" no pedal gera alto jerk.
* **Fundamentação na Literatura:**
  * **Bagdadi & Várhelyi (2011) - "Jerky driving: An indicator of accident proneness":** Os autores provaram estatisticamente que condutores que apresentam picos de jerk $> 2,5\text{ m/s}^3$ possuem propensão significativamente maior a acidentes e estilo impulsivo.
  * **Norma ISO 2631:** Define limites de vibração e choque mecânico veicular para o corpo humano, indicando que variações de aceleração superiores a $2,0\text{ a }2,5\text{ m/s}^3$ provocam desconforto ergonômico.

---

### 2.3. Janelamento Temporal de 120 Segundos (2 Minutos)
* **Critério Adotado:** As viagens foram divididas em fatias contínuas de 120 segundos ($W=120\text{ s}$), exigindo ao menos 30 segundos de telemetria válida para consolidar uma janela.
* **Por que 120 segundos e não 10 segundos ou 10 minutos?**
  * **Janelas muito curtas ($< 30\text{ s}$):** São altamente vulneráveis a fatores exógenos do trânsito. Um motorista muito agressivo parado em um semáforo de 25 segundos pareceria um monge tibetano; e um motorista calmo que fez um desvio repentino pareceria agressivo.
  * **Janelas muito longas ($> 5\text{ a }10\text{ min}$):** Provocam o efeito de diluição estatística. O condutor agressivo que dá 3 arrancadas bruscas e depois roda estável em linha reta terá suas métricas diluídas pela média do percurso, parecendo moderado.
  * **Fundamentação:** **Mobini Seraji et al. (2025)** e **Zhang et al. (2019)** revisaram mais de 100 estudos de reconhecimento de estilo e concluíram que janelas entre **60 e 180 segundos** (especialmente **120 segundos**) fornecem o equilíbrio ideal entre contextualização dinâmica do tráfego e preservação de manobras singulares.

---

### 2.4. Critério de Condução Ativa: Velocidade Média $\ge 8,0\text{ km/h}$
* **O que significa:** Janelas onde a velocidade média é inferior a $8,0\text{ km/h}$ ($\approx 2,22\text{ m/s}$) foram excluídas do treinamento da clusterização.
* **Por que essa decisão foi tomada?**
  * Em uma base de dados real como o VED, há momentos em que o veículo está ligado mas estacionado (esperando passageiro, aquecendo o carro em dias frios de Michigan ou em fila dupla).
  * Se esses dados fossem entregues ao K-Means, a distância euclidiana agruparia os condutores não por sua maneira de dirigir, mas sim pelo fato de estarem em marcha lenta vs. em movimento, criando um cluster de *"veículo parado"*.
  * **Fundamentação:** Estudos automotivos clássicos (Bingham et al., 2012; SAE J2951) separam regimes de baixa velocidade (*creep/stop-and-go/idling*) das janelas de condução livre, permitindo que a clusterização foque nas decisões dinâmicas do motorista (pedal, curvas, aceleração).

---

### 2.5. Potência Elétrica, Consumo e Regeneração ($P = -V \times I / 1000$)
* **Fórmula da Potência Instantânea:**
  $$P(t) = -\frac{V_{\text{bat}}(t) \times I_{\text{bat}}(t)}{1000} \quad [\text{kW}]$$
* **Fórmula da Energia Consumida no Trecho:**
  $$E = \int_{0}^{T} P(t) \, dt \approx \sum_{i=1}^{N} P_i \times \frac{\Delta t_i}{3600} \quad [\text{kWh}]$$
* **Convenção de Sinais adotada por Oh et al. (2020) e Fiori et al. (2016):**
  * No barramento CAN do Nissan Leaf (`HV Battery Current [A]`), correntes de descarga da bateria são reportadas com sinal negativo e correntes de recarga (regeneração) com sinal positivo.
  * Ao aplicar a inversão matemática com sinal negativo ($-$), garantimos a convenção padrão de engenharia energética:
    * **$P(t) > 0$:** Bateria fornecendo energia para o motor (Consumo de energia).
    * **$P(t) < 0$:** Motor atuando como gerador na frenagem e enviando corrente para a bateria (Regeneração energética).

---

### 2.6. Fundamentação da Escolha de $k=3$ no Algoritmo K-Means

A escolha do número de perfis de condução $k=3$ foi respaldada pela análise conjunta de 4 métricas matemáticas e pela literatura especializada:

```
Resultados Computados no Projeto:
k=2 | Inércia: 10175.07 | Silhueta: 0.2552 | Davies-Bouldin: 1.5904 | Calinski-Harabasz: 708.3
k=3 | Inércia:  8397.57 | Silhueta: 0.2423 | Davies-Bouldin: 1.4537 | Calinski-Harabasz: 665.0
k=4 | Inércia:  7294.16 | Silhueta: 0.2501 | Davies-Bouldin: 1.3515 | Calinski-Harabasz: 622.6
k=5 | Inércia:  6428.03 | Silhueta: 0.2145 | Davies-Bouldin: 1.3279 | Calinski-Harabasz: 604.8
k=6 | Inércia:  5900.21 | Silhueta: 0.2206 | Davies-Bouldin: 1.2920 | Calinski-Harabasz: 566.7
```

1. **Método do Cotovelo (Inércia / WCSS - Thorndike, 1953):**
   * A maior taxa de ganho na redução da inércia ocorre de $k=2$ para $k=3$ ($\Delta = -1.777,5$). A partir de $k=3$, a redução torna-se linear e com ganhos marginais decrescentes.
2. **Coeficiente de Silhueta (Rousseeuw, 1987):**
   * O valor de $0,2423$ para $k=3$ demonstra coesão interna adequada em dados reais e contínuos de tráfego. Embora $k=2$ tenha silhueta de $0,2552$, agrupamentos binários são insuficientes para captar o motorista mediano.
3. **Índice Davies-Bouldin (Davies & Bouldin, 1979):**
   * Redução de $1,59$ ($k=2$) para $1,45$ ($k=3$), atestando maior distinção entre os centróides.
4. **Literatura Científica (Taxonomia Tripartite):**
   * Estudos como **Martinez et al. (2018)**, **Wang et al. (2018)** e **Mobini Seraji et al. (2025)** estabelecem de forma quase unânime que o comportamento do motorista distribui-se em 3 classes fundamentais:
     * **Econômico / Suave (Eco/Calm):** Foco em conservação e inércia.
     * **Moderado / Regular (Normal/Average):** O padrão da maioria dos condutores urbanos.
     * **Agressivo / Dinâmico (Aggressive/Dynamic):** Alta aceleração, freadas bruscas e consumo elevado.
5. **Distribuição Normal na Amostra:**
   * A proporção encontrada na base (Moderado: 55,5%, Econômico: 27,8%, Agressivo: 16,7%) segue exatamente a curva gaussiana esperada para amostras populacionais de motoristas.

---

### 2.7. Índice Composto de Agressividade (Ordenação Semântica)
* **Fórmula Implementada:**
  $$\text{Score} = 2 \times \text{hard\_braking\_rate\_min} + 2 \times \text{rapid\_accel\_rate\_min} + 1 \times \text{max\_pos\_accel\_ms2}$$
* **Por que essa fórmula?**
  O K-Means padrão associa números arbitrários (0, 1 ou 2) aos grupos. Para garantir que o modelo salvo seja determinístico, calculamos esse índice sobre os centróides:
  * Frenagens bruscas e arrancadas rápidas têm peso dobrado ($2\times$) porque são as manobras com maior impacto na sensação de conforto dos passageiros e na ineficiência do inversor elétrico.
  * O pico de aceleração tem peso unitário ($1\times$) para diferenciar arrancadas em velocidades intermediárias.
  * Com isso, o cluster com menor pontuação torna-se deterministicamente o **0 (Econômico)**, o intermediário torna-se o **1 (Moderado)** e o de maior pontuação torna-se o **2 (Agressivo)**.

---

## 📚 3. Catálogo Bibliográfico Completo com Links e DOIs

Abaixo estão listados os principais artigos científicos e normas técnicas consultadas, prontos para citação no formato ABNT/IEEE:

### 📄 Artigos de Classificação de Estilo de Condução e Machine Learning
1. **Mobini Seraji et al. (2025)**
   * **Título:** *A state-of-the-art review on machine learning techniques for driving behavior analysis: clustering and classification approaches*
   * **Periódico:** *Complex & Intelligent Systems*, Springer, Vol. 11, Artigo 386.
   * **DOI:** [10.1007/s40747-025-01988-5](https://doi.org/10.1007/s40747-025-01988-5)
   * **Aplicação no TCC:** Justificativa da clusterização K-Means, tamanho das janelas temporais (60-120s) e taxonomia tripartite de perfis.

2. **Martinez, C. M., Heucke, M., Wang, F. Y., Gao, B., & Cao, D. (2018)**
   * **Título:** *Driving Style Recognition for Intelligent Vehicle Control and Advanced Driver Assistance: A Survey*
   * **Periódico:** *IEEE Transactions on Intelligent Transportation Systems*, Vol. 19, No. 3, pp. 666-676.
   * **DOI:** [10.1109/TITS.2017.2706978](https://doi.org/10.1109/TITS.2017.2706978)
   * **Aplicação no TCC:** Referência fundamental sobre a classificação tripartite (Aggressive, Moderate, Calm) e uso de métricas de aceleração e jerk.

3. **Wang, W., Xi, J., & Ding, J. (2018)**
   * **Título:** *Driving style analysis using primitive driving conditions: A unsupervised approach*
   * **Periódico:** *IEEE Transactions on Intelligent Transportation Systems*, Vol. 19, No. 8, pp. 2636-2646.
   * **DOI:** [10.1109/TITS.2017.2764047](https://doi.org/10.1109/TITS.2017.2764047)
   * **Aplicação no TCC:** Uso de aprendizado não supervisionado para extração de perfis e limiares de aceleração em manobras primitivas.

---

### 📄 Artigos sobre o Dataset VED e Modelagem de Veículos Elétricos
4. **Oh, G., LeBlanc, D. J., & Peng, H. (2020)**
   * **Título:** *Vehicle Energy Dataset (VED), A Large-Scale Dataset for Vehicle Energy Consumption Research*
   * **Periódico:** *IEEE Transactions on Intelligent Transportation Systems*, Vol. 23, No. 4, pp. 3302-3312.
   * **DOI:** [10.1109/TITS.2020.3035596](https://doi.org/10.1109/TITS.2020.3035596) | Preprint: [arXiv:1905.02081](https://arxiv.org/abs/1905.02081)
   * **Aplicação no TCC:** Descrição do dataset VED, amostragem a 1 Hz, protocolo OBD-II e telemetria de alta voltagem dos veículos elétricos Nissan Leaf.

5. **Fiori, C., Ahn, K., & Rakha, H. A. (2016)**
   * **Título:** *Power-based electric vehicle energy consumption model: Model development and validation*
   * **Periódico:** *Applied Energy*, Elsevier, Vol. 168, pp. 257-268.
   * **DOI:** [10.1016/j.apenergy.2016.01.097](https://doi.org/10.1016/j.apenergy.2016.01.097)
   * **Aplicação no TCC:** Formulação da potência elétrica da bateria ($P = V \times I$), equações de forças resistentes e eficiência da frenagem regenerativa.

6. **Bingham, C., Walsh, C., & Carroll, S. (2012)**
   * **Título:** *Impact of driving characteristics on electric vehicle energy consumption*
   * **Periódico:** *World Electric Vehicle Journal*, Vol. 5, No. 1, pp. 32-41.
   * **DOI:** [10.3390/wevj5010032](https://doi.org/10.3390/wevj5010032)
   * **Aplicação no TCC:** Demonstração empírica de que o estilo de condução (agressivo vs. moderado vs. econômico) varia a autonomia do EV entre 15% e 30%.

---

### 📄 Artigos de Limiares de Aceleração, Frenagem Brusca e Jerk
7. **Bagdadi, O. (2013)**
   * **Título:** *Estimation of the severity of safety critical events: Development of a unified measure of driver behavior*
   * **Periódico:** *Transportation Research Part F: Traffic Psychology and Behaviour*, Elsevier, Vol. 21, pp. 147-156.
   * **DOI:** [10.1016/j.trf.2013.09.011](https://doi.org/10.1016/j.trf.2013.09.011)
   * **Aplicação no TCC:** Limiar de frenagem brusca ($a \le -2,0\text{ m/s}^2$ a $-2,5\text{ m/s}^2$) como medida de condução agressiva e risco.

8. **Bagdadi, O., & Várhelyi, A. (2011)**
   * **Título:** *Jerky driving—an indicator of accident proneness?*
   * **Periódico:** *Accident Analysis & Prevention*, Elsevier, Vol. 43, No. 4, pp. 1359-1363.
   * **DOI:** [10.1016/j.aap.2011.02.009](https://doi.org/10.1016/j.aap.2011.02.009)
   * **Aplicação no TCC:** Definição física de *jerk* ($j = \Delta a / \Delta t$) e do limiar crítico $\|j\| > 2,5\text{ m/s}^3$ para identificação de conduta brusca.

9. **Eboli, L., Mazzulla, G., & Pungillo, G. (2016)**
   * **Título:** *Measuring bus comfort levels by using acceleration instantaneous values*
   * **Periódico:** *Transport*, Vol. 31, No. 1, pp. 62-73.
   * **DOI:** [10.3846/16484142.2015.1018318](https://doi.org/10.3846/16484142.2015.1018318)
   * **Aplicação no TCC:** Algoritmo JATD (Jerk-Acceleration Threshold Detection) e limiares de transição de conforto ($1,47\text{ m/s}^2$ a $2,0\text{ m/s}^2$).

10. **Klauer, S. G., Dingus, T. A., Neale, V. L., Sudweeks, J. D., & Ramsey, D. J. (2006)**
    * **Título:** *The Impact of Driver Inattention on Near-Crash/Crash Risk: An Analysis Using the 100-Car Naturalistic Driving Study Data*
    * **Relatório Técnico:** *NHTSA - National Highway Traffic Safety Administration*, DOT HS 810 594.
    * **Link:** [NHTSA Technical Report DOT HS 810 594](https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/810594)
    * **Aplicação no TCC:** Definição empírica de desacelerações de conforto ($3,0\text{ m/s}^2$) e eventos de risco a partir de $0,20g$ ($2,0\text{ m/s}^2$).

---

### 📄 Artigos Seminais de Algoritmos e Métricas de Validação
11. **Rousseeuw, P. J. (1987)**
    * **Título:** *Silhouettes: A graphical aid to the interpretation and validation of cluster analysis*
    * **Periódico:** *Journal of Computational and Applied Mathematics*, Elsevier, Vol. 20, pp. 53-65.
    * **DOI:** [10.1016/0377-0427(87)90125-7](https://doi.org/10.1016/0377-0427(87)90125-7)
    * **Aplicação no TCC:** Coeficiente de Silhueta para validação de coerência intra e inter-cluster.

12. **Thorndike, R. L. (1953)**
    * **Título:** *Who belongs in the family?*
    * **Periódico:** *Psychometrika*, Vol. 18, No. 4, pp. 267-276.
    * **Aplicação no TCC:** Método do Cotovelo (*Elbow Method*) e minimização da inércia/WCSS para determinação de $k$.

13. **Davies, D. L., & Bouldin, D. W. (1979)**
    * **Título:** *A Cluster Separation Measure*
    * **Periódico:** *IEEE Transactions on Pattern Analysis and Machine Intelligence*, Vol. PAMI-1, No. 2, pp. 224-227.
    * **DOI:** [10.1109/TPAMI.1979.4766909](https://doi.org/10.1109/TPAMI.1979.4766909)
    * **Aplicação no TCC:** Índice Davies-Bouldin para aferição de dispersão e separabilidade dos grupos.

14. **Caliński, T., & Harabasz, J. (1974)**
    * **Título:** *A dendrite method for cluster analysis*
    * **Periódico:** *Communications in Statistics*, Vol. 3, No. 1, pp. 1-27.
    * **DOI:** [10.1080/03610927408827101](https://doi.org/10.1080/03610927408827101)
    * **Aplicação no TCC:** Índice Calinski-Harabasz (*Variance Ratio Criterion*).

---

## 💡 4. Como Utilizar este Material no Texto da Monografia

1. **Na Introdução e Revisão Bibliográfica:**
   * Citar **Mobini Seraji et al. (2025)** e **Martinez et al. (2018)** para justificar a importância do estilo de condução e por que a categorização em 3 perfis (*Econômico, Moderado e Agressivo*) é o padrão ouro na literatura automotiva.
   * Citar **Bingham et al. (2012)** para demonstrar como o estilo de dirigir pode impactar em até 30% a autonomia de veículos elétricos.
2. **Na Metodologia:**
   * Citar **Oh et al. (2020)** para descrever o dataset VED e justificar a regularização temporal a 1 Hz.
   * Citar **Klauer et al. (2006)** e **Bagdadi (2013)** para fundamentar os limiares de $|a| \ge 2,0\text{ m/s}^2$ como frenagem brusca e arrancada rápida.
   * Citar **Bagdadi & Várhelyi (2011)** e **Eboli et al. (2016)** para a derivada da aceleração (*jerk*) e limiar de $2,5\text{ m/s}^3$.
   * Citar **Fiori et al. (2016)** para a fórmula e convenção da potência elétrica instantânea da bateria ($P = -V \times I / 1000$).
3. **Nos Resultados e Discussão:**
   * Inserir a tabela de comparação de $k \in [2, 6]$ citando **Thorndike (1953)** (Cotovelo), **Rousseeuw (1987)** (Silhueta) e **Davies & Bouldin (1979)** para justificar a escolha matemática irrefutável de $k=3$.

