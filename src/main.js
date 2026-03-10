import { Game } from './Game.js';
import { SimulationRunner } from './SimulationRunner.js';
import { FlappyEnv } from './env/FlappyEnv.js';
import { QLearningAgent } from './agents/QLearningAgent.js';
import { Telemetry } from './utils/Telemetry.js';
import { Dashboard } from './ui/Dashboard.js';

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
    } else {
        startAIMode();
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
}

function startHumanMode() {
    humanGame = new Game(canvas);

    // Maintain speed selection
    const activeSpeed = document.querySelector('.speed-btn.active');
    if (activeSpeed) humanGame.setSpeed(parseInt(activeSpeed.dataset.speed, 10));

    humanGame.start();
}

function startAIMode() {
    // 1. Initialize RL components
    env = new FlappyEnv();
    agent = new QLearningAgent(); // Fresh agent to watch it learn
    telemetry = new Telemetry(100);
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
            telemetry.addEpisode(summary.totalReward, summary.score, agent.epsilon, agent.statesVisited);
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
