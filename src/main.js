import { Game } from './Game.js';
import { SimulationRunner } from './SimulationRunner.js';
import { FlappyEnv } from './env/FlappyEnv.js';
import { QLearningAgent } from './agents/QLearningAgent.js';
import { DQNAgent } from './agents/DQNAgent.js';
import { Telemetry } from './utils/Telemetry.js';
import { Dashboard } from './ui/Dashboard.js';
import { Evaluator } from './utils/Evaluator.js';

// ── UI Elements ──────────────────────────────────────────
const canvas = document.getElementById('canvas');
const modeSelect = document.getElementById('agent-select');
const speedBtns = document.querySelectorAll('.speed-btn');

// ── Global State ─────────────────────────────────────────
let humanGame = null;
let aiRunner = null;
let env = null;
let agent = null;
let telemetry = null;
let dashboard = null;

// Initialize Human Mode by default
startHumanMode();

// ── Mode Switcher ────────────────────────────────────────
modeSelect.addEventListener('change', (e) => {
    stopCurrentModes();
    if (e.target.value === 'human') {
        startHumanMode();
    } else if (e.target.value === 'benchmark') {
        startBenchmarkMode();
    } else {
        startAIMode(e.target.value);
    }
});

// ── Speed Controls ───────────────────────────────────────
speedBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
        const speed = parseInt(btn.dataset.speed, 10);

        if (humanGame) humanGame.setSpeed(speed);
        if (aiRunner) aiRunner.setStepsPerFrame(speed);

        speedBtns.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
    });
});

// ── Logic ───────────────────────────────────────────────

function stopCurrentModes() {
    if (humanGame) {
        humanGame.stop();
        humanGame = null;
    }
    if (aiRunner) {
        aiRunner.stop();
        aiRunner = null;
    }
    if (dashboard) {
        dashboard.hide();
    }
    document.getElementById('benchmark-dashboard').classList.add('hidden');
    document.getElementById('canvas').style.display = 'block';
}

function startHumanMode() {
    humanGame = new Game(canvas);

    // Maintain speed selection
    const activeSpeed = document.querySelector('.speed-btn.active');
    if (activeSpeed) humanGame.setSpeed(parseInt(activeSpeed.dataset.speed, 10));

    humanGame.start();
}

function startAIMode(agentType) {
    // 1. Initialize RL components
    env = new FlappyEnv();

    if (agentType === 'dqn') {
        agent = new DQNAgent();
    } else {
        agent = new QLearningAgent();
    }

    // Smooth moving average: Q-Learning is noisy (100 window), DQN is smoother (50 window)
    telemetry = new Telemetry(agentType === 'dqn' ? 50 : 100);
    dashboard = new Dashboard(telemetry);
    dashboard.show();

    // 2. We use Game as a pure renderer for the AI
    const renderer = new Game(canvas);

    // 3. Runner orchestrates training + rendering
    let lastState = null;
    let lastAction = null;

    aiRunner = new SimulationRunner(env, {
        actionProvider: (state) => {
            const action = agent.act(state);
            lastState = state;
            lastAction = action;
            return action;
        },
        renderer: (e) => renderer.renderEnv(e),
        onStep: (result) => {
            // TD Update during play!
            agent.learn(lastState, lastAction, result.reward, result.state, result.done);
        },
        onEpisodeEnd: (summary) => {
            agent.endEpisode();
            // Record metrics
            telemetry.addEpisode(summary.totalReward, summary.score, agent.epsilon, agent.statesVisited || "N/A");
            // Update UI
            dashboard.update();

            // Stop training at 50,000 episodes
            if (telemetry.episodes >= 50000) {
                aiRunner.stop();
                console.log("Training complete (50k episodes reached).");
            }
        }
    });

    // Maintain speed selection
    const activeSpeed = document.querySelector('.speed-btn.active');
    if (activeSpeed) aiRunner.setStepsPerFrame(parseInt(activeSpeed.dataset.speed, 10));

    // Force first render frame to GameState.play so "Get Ready" never flashes
    renderer.renderEnv(env);

    // Let's use incrementing seeds so it experiences different pipes
    let seedCount = 42;
    aiRunner._onReset = () => { env.seed = seedCount++; };
    aiRunner.startVisual(seedCount++);
}

async function startBenchmarkMode() {
    // Hide game canvas since it will run headless in browser
    document.getElementById('canvas').style.display = 'none';
    const bDash = document.getElementById('benchmark-dashboard');
    bDash.classList.remove('hidden');
    
    document.getElementById('bm-agent-name').innerText = 'Cargando modelos desde /models/...';
    document.getElementById('bm-progress-bar').style.width = '0%';
    document.getElementById('bm-table-body').innerHTML = '';
    document.getElementById('bm-conclusion').innerText = '';

    try {
        const qAgent = new QLearningAgent();
        const dqnAgent = new DQNAgent();

        // 1. Load models explicitly
        const qReq = await fetch('./models/qlearning.json');
        if (!qReq.ok) throw new Error("Q-Learning no encontrado. Corre 'node src/examples/benchmark_agents.js' primero");
        const qData = await qReq.json();
        qAgent.load(qData);

        await dqnAgent.load('./models/dqn/model.json'); // Fetches implicitly

        const agents = [
            { name: 'Q-Learning', agent: qAgent },
            { name: 'DQN (TF.js)', agent: dqnAgent }
        ];

        // 2. Start Evaluator
        const env = new FlappyEnv();
        const evaluator = new Evaluator(env, agents);

        const results = await evaluator.evaluateAllAsync({ episodes: 200, startSeed: 424242 }, (agentName, progress, stats) => {
            document.getElementById('bm-agent-name').innerText = `Evaluando: ${agentName}`;
            document.getElementById('bm-progress-bar').style.width = `${Math.floor(progress * 100)}%`;
            document.getElementById('bm-stats').innerText = `Avg: ${stats.avgScore} | Max: ${stats.maxScore}`;
        });

        // 3. Render Results
        document.getElementById('bm-agent-name').innerText = 'Benchmark Finalizado';
        document.getElementById('bm-progress-bar').style.width = '100%';
        document.getElementById('bm-stats').innerText = '';

        const tBody = document.getElementById('bm-table-body');
        tBody.innerHTML = '';
        for (const [agentName, metrics] of Object.entries(results)) {
            tBody.innerHTML += `
                <tr style="border-bottom: 1px solid #3d3d5c;">
                    <td style="padding: 5px;">${agentName}</td>
                    <td style="padding: 5px;">${metrics.avgScore}</td>
                    <td style="padding: 5px;">${metrics.maxScore}</td>
                    <td style="padding: 5px;">${metrics.stdDev}</td>
                </tr>
            `;
        }

        const qAvg = results['Q-Learning'].avgScore;
        const dAvg = results['DQN (TF.js)'].avgScore;

        if (qAvg > dAvg) {
            document.getElementById('bm-conclusion').innerText = `✅ Q-Learning rinde mejor por +${(qAvg - dAvg).toFixed(2)} Avg Score`;
        } else if (dAvg > qAvg) {
            document.getElementById('bm-conclusion').innerText = `✅ DQN rinde mejor por +${(dAvg - qAvg).toFixed(2)} Avg Score`;
        } else {
            document.getElementById('bm-conclusion').innerText = `🤝 Empate Estadístico`;
        }

    } catch(err) {
        document.getElementById('bm-agent-name').innerText = 'Error al ejecutar benchmark';
        document.getElementById('bm-stats').innerText = err.message;
        console.error("Benchmark UI Error:", err);
    }
}

