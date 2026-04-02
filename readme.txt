==========================================================================
UNIVERSIDADE DO PORTO - FACULDADE DE CIÊNCIAS (FCUP)
CC3046 - Introdução à Robótica Inteligente | Ano Letivo 2025/2026
PROJETO: Vision-Based Autonomous Pursuit-Evasion in Urban Environments
==========================================================================

GRUPO: T3-G9
ESTUDANTES:
- Guilherme Klippel (202300276)
- Pedro Paulo Basilio (202300926)
- Yan Coelho (202300916)

--------------------------------------------------------------------------
1. RESUMO DO PROJETO
--------------------------------------------------------------------------
Este projeto explora técnicas de Reinforcement Learning (RL), especificamente 
Proximal Policy Optimization (PPO), para treinar um veículo evadido num 
cenário de "cops and robbers" dentro do simulador Webots. 

O objetivo principal é que o evadido aprenda a navegar em ambientes urbanos 
dinâmicos, utilizando apenas sensores de bordo para quebrar a linha de visão 
(Line-of-Sight) e evitar a captura por um agente perseguidor.

--------------------------------------------------------------------------
2. ABORDAGEM TÉCNICA
--------------------------------------------------------------------------
- Algoritmo Base: PPO (Proximal Policy Optimization) para controlo contínuo.
- Memória Temporal: Camadas LSTM para lidar com observabilidade parcial e 
  perda temporária de sinal visual do perseguidor.
- Espaço de Estados (Sensores):
  * Câmera: Deteção visual do perseguidor.
  * LiDAR: Percepção de obstáculos e infraestrutura urbana.
  * GPS: Utilizado pelo perseguidor (com frequência X) e para métricas.
- Espaço de Ação (Atuadores):
  * Steering: Ângulo de direção das rodas frontais.
  * Throttle: Intensidade de aceleração e travagem.

--------------------------------------------------------------------------
3. REQUISITOS E INSTALAÇÃO
--------------------------------------------------------------------------
- Simulador: Webots R2023b ou superior.
- Linguagem: Python 3.10.x.
- Dependências: stable-baselines3[extra], sb3-contrib, torch.

Instalação rápida:
1. Criar venv: python -m venv .venv
2. Ativar venv: .venv\Scripts\activate
3. Instalar bibliotecas: pip install -r requirements.txt

--------------------------------------------------------------------------
4. ESTRUTURA DE DIRETÓRIOS
--------------------------------------------------------------------------
/controllers/evader_controller/  -> Código fonte do agente evadido.
/worlds/                         -> Ficheiros .wbt (Cena "City Traffic").
/models/                         -> (Futuro) Modelos PPO+LSTM treinados.
README.txt                       -> Documentação atual.
requirements.txt                 -> Lista de dependências Python.

--------------------------------------------------------------------------
5. CRONOGRAMA DE DESENVOLVIMENTO (MILESTONES)
--------------------------------------------------------------------------
- Semanas 1-2: Setup da simulação e testes de sensores. (CONCLUÍDO)
- Semanas 3-5: Implementação do PPO e comportamento base do perseguidor.
- Semanas 6-7: Integração de LSTM e refinamento da Reward Function.
- Semanas 8-10: Experiências e recolha de métricas (Survival Time/Escape Rate).
- Semanas 11-12: Finalização do relatório e demonstração final.

--------------------------------------------------------------------------
6. EXECUÇÃO
--------------------------------------------------------------------------
1. Abrir o Webots com o mundo "my_city_traffic.wbt".
2. Colocar o controlador do veículo em modo <extern>.
3. Executar evader_controller.py a partir do IDE (PyCharm).
==========================================================================