import { FlappyEnv } from '../env/FlappyEnv.js';
import { QLearningAgent } from '../agents/QLearningAgent.js';

const env = new FlappyEnv();
const agent = new QLearningAgent();

let state = env.reset(42);
console.log("Initial state:", state, "Discrete:", agent._discretizer.discretize(state).key);

for (let i = 0; i < 20; i++) {
    const action = agent.act(state);
    const step = env.step(action);
    console.log(`Step ${i} | a=${action} | s=${step.state} | key=${agent._discretizer.discretize(step.state).key}`);
    state = step.state;
}
