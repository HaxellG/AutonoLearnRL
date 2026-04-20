// src/ui/Dashboard.js
export class Dashboard {
    constructor(telemetry) {
        this.telemetry = telemetry;

        // DOM Elements
        this.container = document.getElementById('dashboard');
        this.lbEpisode = document.getElementById('lb-episode');
        this.lbEpsilon = document.getElementById('lb-epsilon');
        this.lbMaxScore = document.getElementById('lb-maxscore');
        this.lbStates = document.getElementById('lb-states');

        this.chartReward = null;
        this.chartScore = null;
        this.lastUpdateEp = 0;
        this.isActive = false;

        this._initCharts();
    }

    show() {
        this.container.classList.remove('hidden');
        this.isActive = true;
    }

    hide() {
        this.container.classList.add('hidden');
        this.isActive = false;
        this.destroy(); // Safely clear charts to prevent UI memory leaks
    }

    /**
     * Clear chart instances when switching agents to avoid canvas reuse errors
     */
    destroy() {
        if (this.chartReward) {
            this.chartReward.destroy();
            this.chartReward = null;
        }
        if (this.chartScore) {
            this.chartScore.destroy();
            this.chartScore = null;
        }
    }

    update() {
        if (!this.isActive) return;

        // Update cards
        this.lbEpisode.innerText = this.telemetry.episodes.toLocaleString();
        this.lbEpsilon.innerText = this.telemetry.currentEpsilon.toFixed(3);
        this.lbMaxScore.innerText = this.telemetry.maxScore.toString();
        this.lbStates.innerText = this.telemetry.statesVisited.toString();

        // 1. We don't want to draw 50,000 points. Subsample to max 500 points.
        const total = this.telemetry.episodes;
        if (total === 0) return;

        // Render every Nth frame to avoid lag 
        // Only update charts if at least 100 episodes have passed since last update
        if (total - this.lastUpdateEp < 100 && total > 100) return;
        this.lastUpdateEp = total;

        const maxPoints = 500;
        const step = Math.max(1, Math.floor(total / maxPoints));

        const labels = [];
        const rewards = [];
        const avgRewards = [];
        const scores = [];
        const avgScores = [];

        for (let i = 0; i < total; i += step) {
            labels.push(this.telemetry.labels[i]);
            rewards.push(this.telemetry.rewards[i]);
            avgRewards.push(this.telemetry.avgRewards[i]);
            scores.push(this.telemetry.scores[i]);
            avgScores.push(this.telemetry.avgScores[i]);
        }

        // Always push the very last point
        if (total > 0 && (total - 1) % step !== 0) {
            const last = total - 1;
            labels.push(this.telemetry.labels[last]);
            rewards.push(this.telemetry.rewards[last]);
            avgRewards.push(this.telemetry.avgRewards[last]);
            scores.push(this.telemetry.scores[last]);
            avgScores.push(this.telemetry.avgScores[last]);
        }

        // Update Reward Chart
        this.chartReward.data.labels = labels;
        this.chartReward.data.datasets[0].data = rewards;
        this.chartReward.data.datasets[1].data = avgRewards;
        this.chartReward.update('none'); // no animation for performance

        // Update Score Chart
        this.chartScore.data.labels = labels;
        this.chartScore.data.datasets[0].data = scores;
        this.chartScore.data.datasets[1].data = avgScores;
        this.chartScore.update('none');
    }

    _initCharts() {
        if (!window.Chart) {
            console.error("Chart.js not loaded. Dashboard disabled.");
            return;
        }

        // Defaults for dark theme
        Chart.defaults.color = '#9ca3af';
        Chart.defaults.font.family = 'sans-serif';

        const ctxReward = document.getElementById('chart-reward').getContext('2d');
        this.chartReward = new Chart(ctxReward, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Reward',
                        data: [],
                        borderColor: 'rgba(48, 192, 223, 0.2)', // light blue faded
                        borderWidth: 1,
                        pointRadius: 0,
                    },
                    {
                        label: 'Moving Avg',
                        data: [],
                        borderColor: '#30c0df', // solid blue
                        borderWidth: 2,
                        pointRadius: 0,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: { display: false },
                    y: { beginAtZero: false } // rewards can be negative
                },
                plugins: {
                    legend: { display: true, position: 'top' }
                }
            }
        });

        const ctxScore = document.getElementById('chart-score').getContext('2d');
        this.chartScore = new Chart(ctxScore, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Score',
                        data: [],
                        borderColor: 'rgba(255, 99, 132, 0.2)', // red faded
                        borderWidth: 1,
                        pointRadius: 0,
                    },
                    {
                        label: 'Moving Avg',
                        data: [],
                        borderColor: '#ff6384', // solid red
                        borderWidth: 2,
                        pointRadius: 0,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: { display: false },
                    y: { beginAtZero: true }
                },
                plugins: {
                    legend: { display: true, position: 'top' }
                }
            }
        });
    }
}
