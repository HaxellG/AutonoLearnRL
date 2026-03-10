import { readFileSync } from 'node:fs';

const data = JSON.parse(readFileSync('qtable.json'));
console.log(`Total states: ${data.qTable.length}`);
for (const [key, q0, q1] of data.qTable) {
    console.log(`State: ${key} | Q0=${q0.toFixed(2)} Q1=${q1.toFixed(2)}`);
}
