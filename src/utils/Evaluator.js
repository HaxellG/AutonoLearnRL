import { SimulationRunner } from '../SimulationRunner.js';

/**
 * Headless evaluator para comparar numéricamente el desempeño de múltiples agentes.
 * Ignora la exploración (`epsilon=0`) durante la evaluación.
 */
export class Evaluator {
    /**
     * @param {import('../env/FlappyEnv.js').FlappyEnv} env
     * @param {Array<{name: string, agent: any}>} agents
     */
    constructor(env, agents) {
        this.env = env;
        this.agents = agents;
    }

    /**
     * Evalúa todos los agentes con las mismas semillas.
     * @param {Object} options
     * @param {number} options.episodes - Número de episodios de evaluación.
     * @param {number} options.startSeed - Semilla inicial (se incrementará).
     * @returns {Object} Diccionario con resultados por agente.
     */
    evaluateAll({ episodes = 100, startSeed = 1000 }) {
        const report = {};

        // Generar lista estática de semillas para justicia total
        const seeds = Array.from({ length: episodes }, (_, i) => startSeed + i);

        for (const { name, agent } of this.agents) {
            // Reemplazar la política normal por "bestAction" paramétrica
            const greedyPolicy = (state) => {
                if (typeof agent.bestAction === 'function') {
                    return agent.bestAction(state);
                }
                // Fallback si no tiene bestAction (debería tenerlo según el plan)
                const oldEpsilon = agent.epsilon;
                agent.epsilon = 0;
                const act = agent.act(state);
                agent.epsilon = oldEpsilon;
                return act;
            };

            const runner = new SimulationRunner(this.env, { actionProvider: greedyPolicy });
            let totalScore = 0;
            let maxScore = 0;
            let minScore = Infinity;
            const scores = [];

            // Evaluamos episodio a episodio para tener control exacto de la semilla
            for (const s of seeds) {
                const results = runner.runSync({ episodes: 1, seed: s });
                const epScore = results[0].score;
                
                totalScore += epScore;
                scores.push(epScore);
                maxScore = Math.max(maxScore, epScore);
                minScore = Math.min(minScore, epScore);
            }

            const avgScore = totalScore / episodes;
            
            // Computar desviación estándar
            const variance = scores.reduce((acc, val) => acc + Math.pow(val - avgScore, 2), 0) / episodes;
            const stdDev = Math.sqrt(variance);

            report[name] = {
                avgScore: Number(avgScore.toFixed(2)),
                maxScore,
                minScore,
                stdDev: Number(stdDev.toFixed(2)),
                episodes
            };
        }

        return report;
    }

    /**
     * Evalúa todos los agentes de forma asíncrona, cediendo control intermitentemente
     * para no bloquear el renderizado de la UI web.
     * @param {Object} options
     * @param {number} options.episodes - Número de episodios de evaluación.
     * @param {number} options.startSeed - Semilla inicial.
     * @param {Function} [progressCallback] - Función `(currentAgentName, progressRatio, metricsSoFar)` llamada frecuentemente.
     * @returns {Promise<Object>} Diccionario con resultados por agente.
     */
    async evaluateAllAsync({ episodes = 100, startSeed = 1000 }, progressCallback) {
        const report = {};
        const seeds = Array.from({ length: episodes }, (_, i) => startSeed + i);

        for (const { name, agent } of this.agents) {
            const greedyPolicy = (state) => {
                if (typeof agent.bestAction === 'function') return agent.bestAction(state);
                const old = agent.epsilon;
                agent.epsilon = 0;
                const act = agent.act(state);
                agent.epsilon = old;
                return act;
            };

            const runner = new SimulationRunner(this.env, { actionProvider: greedyPolicy });
            let totalScore = 0;
            let maxScore = 0;
            let minScore = Infinity;
            const scores = [];

            for (let i = 0; i < seeds.length; i++) {
                // Async headless run for 1 episode
                const results = await runner.runHeadless({ episodes: 1, seed: seeds[i], yieldEvery: 1000 });
                const epScore = results[0].score;
                
                totalScore += epScore;
                scores.push(epScore);
                maxScore = Math.max(maxScore, epScore);
                minScore = Math.min(minScore, epScore);

                if (progressCallback) {
                    progressCallback(name, (i + 1) / episodes, {
                        avgScore: (totalScore / (i + 1)).toFixed(2),
                        maxScore
                    });
                }
                
                // IMPORTANT: Force browser to paint UI progress by yielding to macro-task queue
                // Otherwise a micro-task chain will freeze the browser loop
                await new Promise(r => setTimeout(r, 0));
            }

            const avgScore = totalScore / episodes;
            const variance = scores.reduce((acc, val) => acc + Math.pow(val - avgScore, 2), 0) / episodes;
            const stdDev = Math.sqrt(variance);

            report[name] = {
                avgScore: Number(avgScore.toFixed(2)),
                maxScore,
                minScore,
                stdDev: Number(stdDev.toFixed(2)),
                episodes
            };
        }

        return report;
    }
}
