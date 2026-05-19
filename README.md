# AutonoLearn RL — Flappy Bird

Aplicación Web de Reinforcement Learning sobre el juego Flappy Bird, implementando y comparando tres agentes:

- **Q-Learning** — Tabular
- **DQN** — Deep Q-Network (red neuronal MLP 3→24→24→2)
- **Double DQN** — Reduce el sesgo de sobreestimación de DQN

---

## Requisitos

| Herramienta | Versión recomendada |
|:---|:---|
| **Node.js** | `v20.x` (requerido por TensorFlow.js C++ backend) |

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

## Lanzar App Web

Utiliza la extensión Live Server para abrir el proyecto en el navegador.

---

## Entrenar los agentes

Cada agente se puede entrenar de forma **independiente**. Los modelos y resultados se guardan automáticamente en `models_final/` y `results/`. Ten en cuenta que la dificultad a la que se entrenan depende del parámetro global `pipe.gap` definido en `src/config.js`, donde 125 equivale a la dificultad normal y 100 a la difícil.

### Q-Learning (tabular)
```bash
npm run train:qlearning
# o directamente:
node src/examples/train_qlearning.js
```

---

### DQN (Deep Q-Network)
```bash
npm run train:dqn
# o directamente:
node src/examples/train_dqn.js
```

---

### Double DQN
```bash
npm run train:ddqn
# o directamente:
node src/examples/train_ddqn.js
```

---

### Experimento completo (los 3 agentes seguidos + Head-to-Head)
```bash
npm run train:all
# o directamente:
node src/examples/full_experiment.js
```
- Duración estimada: **~3 o más horas**

> ⚠️ No se recomienda correr `train:all` si el sistema tiene menos de 8-10 GB de RAM libre, ya que DQN y DDQN se ejecutan secuencialmente pero ambos cargan TensorFlow en el mismo proceso.

---

## Configuración del entorno

Los parámetros principales están centralizados en `src/config.js`:

| Parámetro | Descripción | Valor actual |
|:---|:---|:---|
| `pipe.gap` | Espacio vertical entre tuberías | `125` px |
| `rewards.survive` | Recompensa por frame supervivido | `+0.1` |
| `rewards.passPipe` | Recompensa por cruzar tubo | `+1.0` |
| `rewards.collision` | Penalización por choque | `-1.0` |

etc... `src/config.js` para más información.

---