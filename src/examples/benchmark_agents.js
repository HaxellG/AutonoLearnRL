import { CONFIG } from '../config.js';
import { FlappyEnv } from '../env/FlappyEnv.js';
import { SimulationRunner } from '../SimulationRunner.js';
import { QLearningAgent } from '../agents/QLearningAgent.js';
import { DQNAgent } from '../agents/DQNAgent.js';
import { Evaluator } from '../utils/Evaluator.js';
import fs from 'fs';
import path from 'path';

import * as tf from '@tensorflow/tfjs';
globalThis.tf = tf;

// Limit testing length so perfect agents don't hang the script indefinitely.
// 5,000 steps is around a score of ~50.
CONFIG.env.maxStepsPerEpisode = 5000; 

const Q_TRAINING_EPISODES = 15000;
const DQN_TRAINING_EPISODES = 2000;
const EVAL_EPISODES = 200;

console.log(`\n🚀 Iniciando AutonoLearn Evaluación Comparativa`);
console.log(`===============================================`);
console.log(`Parámetros de la prueba:`);
console.log(`- Capping de pasos por ep.: ${CONFIG.env.maxStepsPerEpisode}`);
console.log(`- Entrenando Q-Learning:    ${Q_TRAINING_EPISODES} Episodios`);
console.log(`- Entrenando DQN (TF.js):   ${DQN_TRAINING_EPISODES} Episodios`);
console.log(`- Fase de Evaluación:       ${EVAL_EPISODES} Episodios por Agente`);
console.log(`===============================================\n`);

async function runBenchmark() {
    // 1. Instanciar Agentes
    const qAgent = new QLearningAgent();
    // Acelerando decaimiento para entrenamiento veloz
    qAgent.epsilonDecay = Math.pow(0.01, 1 / (Q_TRAINING_EPISODES * 0.8)); 

    const dqnAgent = new DQNAgent();
    dqnAgent.epsilonDecay = Math.pow(0.01, 1 / (DQN_TRAINING_EPISODES * 0.8));

    const agents = [
        { name: 'Q-Learning', agent: qAgent, episodes: Q_TRAINING_EPISODES },
        { name: 'DQN (TF.js)', agent: dqnAgent, episodes: DQN_TRAINING_EPISODES }
    ];

    // 2. Headless Training
    console.log(`🔧 Fase 1: Entrenamiento Acelerado...`);
    const env = new FlappyEnv();
    
    for (const { name, agent, episodes } of agents) {
        process.stdout.write(`   Entrenando ${name} / ${episodes} ep... `);
        const startTime = performance.now();
        
        let lastState, lastAction;
        const runner = new SimulationRunner(env, {
            actionProvider: (state) => {
                const action = agent.act(state);
                lastState = state;
                lastAction = action;
                return action;
            },
            onStep: (result) => agent.learn(lastState, lastAction, result.reward, result.state, result.done),
            onEpisodeEnd: () => agent.endEpisode()
        });

        if (name === 'DQN (TF.js)') {
            // yieldEvery reducido a 32 para permitir que las promesas asíncronas
            // de trainOnBatch se envíen y no se asfixie el modelo en Node.
            await runner.runHeadless({ episodes, yieldEvery: 32 });
        } else {
            // runSync modificado ahora llama a onStep correctamente.
            runner.runSync({ episodes });
        }
        
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
        console.log(`[Hecho en ${elapsed}s]`);

        // Save agent
        const modelsDir = path.join(process.cwd(), 'models');
        if (!fs.existsSync(modelsDir)) fs.mkdirSync(modelsDir, { recursive: true });

        if (name === 'Q-Learning') {
            fs.writeFileSync(path.join(modelsDir, 'qlearning.json'), JSON.stringify(agent.save(), null, 2));
        } else if (name === 'DQN (TF.js)') {
            const dqnDir = path.join(modelsDir, 'dqn');
            if (!fs.existsSync(dqnDir)) fs.mkdirSync(dqnDir, { recursive: true });
            
            await agent.save(tf.io.withSaveHandler(async (artifacts) => {
                const modelJSON = {
                    modelTopology: artifacts.modelTopology,
                    format: artifacts.format,
                    generatedBy: artifacts.generatedBy,
                    convertedBy: artifacts.convertedBy,
                    weightsManifest: [{
                        paths: ['weights.bin'],
                        weights: artifacts.weightSpecs
                    }]
                };
                fs.writeFileSync(path.join(dqnDir, 'model.json'), JSON.stringify(modelJSON));
                if (artifacts.weightData) {
                    fs.writeFileSync(path.join(dqnDir, 'weights.bin'), Buffer.from(artifacts.weightData));
                }
                return {
                    modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' }
                };
            }));
        }
        console.log(`   💾 Modelo guardado en /models/`);
    }

    // 3. Evaluation
    console.log(`\n📊 Fase 2: Evaluación Clínica (${EVAL_EPISODES} Episodios con misma serie de semillas)...`);
    const evalEnv = new FlappyEnv();
    const evaluator = new Evaluator(evalEnv, agents);
    
    const results = evaluator.evaluateAll({ episodes: EVAL_EPISODES, startSeed: 0 });

    // 4. Reporte
    console.log(`\n🏆 Resultados Comparativos:`);
    console.table(results);

    // Comparativa directa
    const qAvg = results['Q-Learning'].avgScore;
    const dAvg = results['DQN (TF.js)'].avgScore;

    console.log(`\nConclusión:`);
    if (qAvg > dAvg) {
        console.log(`✅ Q-Learning superó a DQN por una diferencia de media de ${(qAvg - dAvg).toFixed(2)} score.`);
    } else if (dAvg > qAvg) {
        console.log(`✅ DQN superó a Q-Learning por una diferencia de media de ${(dAvg - qAvg).toFixed(2)} score.`);
    } else {
        console.log(`🤝 Empate estadístico en score promedio.`);
    }
}

runBenchmark().catch(console.error);
