/**
 * Telemetry Logger
 * 
 * Collects episodic metrics (Reward, Score, Epsilon) during training.
 * Provides rolling averages and raw data arrays suitable for Chart.js.
 */
export class Telemetry {
    constructor(movingAverageWindow = 100) {
        this.maWindow = movingAverageWindow;

        this.episodes = 0;
        this.maxScore = 0;
        this.statesVisited = 0;
        this.currentEpsilon = 1.0;

        // Raw history
        this.rewards = [];
        this.scores = [];
        this.epsilons = [];

        // Derived data for charts
        this.labels = []; // X-axis (episode numbers)
        this.avgRewards = []; // Y-axis 1
        this.avgScores = []; // Y-axis 2
    }

    /**
     * Call at the end of every episode to push metrics.
     */
    addEpisode(reward, score, epsilon, statesVisited) {
        this.episodes++;
        this.currentEpsilon = epsilon;
        this.statesVisited = statesVisited;

        if (score > this.maxScore) {
            this.maxScore = score;
        }

        this.rewards.push(reward);
        this.scores.push(score);
        this.epsilons.push(epsilon);
        this.labels.push(this.episodes);

        // Calculate moving averages for smoother charts
        const start = Math.max(0, this.rewards.length - this.maWindow);

        // Sums
        let sumR = 0;
        let sumS = 0;
        for (let i = start; i < this.rewards.length; i++) {
            sumR += this.rewards[i];
            sumS += this.scores[i];
        }

        const count = this.rewards.length - start;
        this.avgRewards.push(sumR / count);
        this.avgScores.push(sumS / count);
    }
}
