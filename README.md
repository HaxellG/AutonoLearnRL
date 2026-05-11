# AutonoLearn RL — Flappy Bird

Entorno de Reinforcement Learning sobre el juego Flappy Bird, implementando y comparando tres agentes:

- **Q-Learning** — Tabular, sin redes neuronales
- **DQN** — Deep Q-Network (red neuronal MLP 3→24→24→2)
- **Double DQN** — Reduce el sesgo de sobreestimación de DQN

---

## Requisitos

| Herramienta | Versión recomendada |
|:---|:---|
| **Node.js** | `v20.x` (requerido por TensorFlow.js C++ backend) |
| **nvm** | cualquiera (para gestionar versiones de Node) |
| **npm** | `v10+` (incluido con Node 20) |

> ⚠️ **Importante:** `@tensorflow/tfjs-node` compila binarios nativos en C++. Solo funciona correctamente con **Node v20**. Versiones superiores o inferiores pueden fallar en la instalación o en tiempo de ejecución.

---

## Instalación

```bash
# 1. Instalar Node v20 con nvm (si no lo tienes)
nvm install 20
nvm use 20

# 2. Clonar el repositorio
git clone https://github.com/HaxellG/AutonoLearn-RL.git
cd AutonoLearn-RL

# 3. Instalar dependencias
npm install
```

---

## Jugar manualmente

Abre `index.html` directamente en el navegador o ejecuta:

```bash
npm start
```

---

## Entrenar los agentes

Cada agente se puede entrenar de forma **independiente**. Los modelos y resultados se guardan automáticamente en `models_final/` y `results/`.

### Q-Learning (tabular)
```bash
npm run train:qlearning
# o directamente:
node src/examples/train_qlearning.js
```
- Episodios: **20,000**
- Duración estimada: ~5s
- Salida: `models_final/qlearning.json`, `results/qlearning_results.json`

---

### DQN (Deep Q-Network)
```bash
npm run train:dqn
# o directamente:
node src/examples/train_dqn.js
```
- Episodios: **20,000**
- Duración estimada: ~45–60 min
- Salida: `models_final/dqn/model.json`, `results/dqn_results.json`

---

### Double DQN
```bash
npm run train:ddqn
# o directamente:
node src/examples/train_ddqn.js
```
- Episodios: **30,000**
- Duración estimada: ~60–90 min
- Salida: `models_final/ddqn/model.json`, `results/ddqn_results.json`

---

### Experimento completo (los 3 agentes seguidos + Head-to-Head)
```bash
npm run train:all
# o directamente:
node src/examples/full_experiment.js
```
- Duración estimada: **~2–3 horas**
- Salida: `models_final/`, `results/experiment_log_final.txt`, `results/experiment_results_final.json`

> ⚠️ No se recomienda correr `train:all` si el sistema tiene menos de 8 GB de RAM libre, ya que DQN y DDQN se ejecutan secuencialmente pero ambos cargan TensorFlow en el mismo proceso.

---

## Estructura del proyecto

```
.
├── index.html                        # Interfaz web del juego
├── style.css                         # Estilos de la UI
├── src/
│   ├── config.js                     # Configuración central (gap, física, recompensas, RL)
│   ├── main.js                       # Entry point de la UI
│   ├── SimulationRunner.js           # Motor de simulación headless
│   ├── agents/
│   │   ├── QLearningAgent.js         # Agente tabular
│   │   ├── DQNAgent.js               # Red neuronal DQN
│   │   ├── DoubleDQNAgent.js         # Red neuronal Double DQN
│   │   ├── ReplayBuffer.js           # Memoria de experiencias
│   │   └── StateDiscretizer.js       # Discretizador para Q-Learning
│   ├── env/
│   │   └── FlappyEnv.js              # Entorno RL (física, recompensas, estado)
│   ├── examples/
│   │   ├── train_qlearning.js        # ← Script individual Q-Learning
│   │   ├── train_dqn.js              # ← Script individual DQN
│   │   ├── train_ddqn.js             # ← Script individual Double DQN
│   │   └── full_experiment.js        # Script completo (los 3 juntos)
│   └── ui/
│       └── Dashboard.js              # Dashboard visual en browser
├── models_final/                     # Modelos entrenados guardados aquí
│   ├── qlearning.json
│   ├── dqn/
│   └── ddqn/
└── results/                          # Logs y JSONs de resultados
    ├── experiment_log_final.txt
    ├── qlearning_results.json
    ├── dqn_results.json
    ├── ddqn_results.json
    └── experiment_results_final.json
```

---

## Configuración del entorno

Los parámetros principales están centralizados en `src/config.js`:

| Parámetro | Descripción | Valor actual |
|:---|:---|:---|
| `pipe.gap` | Espacio vertical entre tuberías | `150` px |
| `rewards.survive` | Recompensa por frame supervivido | `+0.1` |
| `rewards.passPipe` | Recompensa por cruzar tubo | `+1.0` |
| `rewards.collision` | Penalización por choque | `-1.0` |
| `rl.qlearning.alpha` | Tasa de aprendizaje tabular | `0.1` |
| `rl.qlearning.gamma` | Factor de descuento | `0.95` |

---

## Hiperparámetros de los agentes

### Q-Learning
| Parámetro | Valor |
|:---|:---|
| alpha (learning rate) | 0.1 |
| gamma | 0.95 |
| epsilon inicial | 1.0 |
| epsilon final | 0.01 |
| Bins de discretización | dx:15, dy:20, vy:15 |

### DQN & Double DQN
| Parámetro | Valor |
|:---|:---|
| learning rate | 0.001 |
| gamma | 0.99 |
| epsilon inicial | 1.0 |
| epsilon final | 0.01 |
| batch size | 64 |
| replay buffer | 10,000 |
| target update freq | 5,000 pasos |
| arquitectura | 3 → 24(ReLU) → 24(ReLU) → 2(Linear) |

---

## Notas técnicas

- El **State Aliasing bug** (ceguera al pasar tuberías) fue corregido mediante `_getActivePipe()` en `FlappyEnv.js`.
- DQN y Double DQN requieren **Node v20** por compatibilidad con el backend C++ de `@tensorflow/tfjs-node`.
- El entrenamiento es **determinista** gracias al PRNG `SeededRandom` — misma semilla = mismo resultado exacto.
