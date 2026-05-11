// src/ui/Dashboard.js
export class Dashboard {
    constructor(telemetry) {
        this.telemetry = telemetry;

        // DOM Elements
        this.container = document.getElementById('dashboard');
        this.lbEpisode = document.getElementById('lb-episode');
        this.lbEpsilon = document.getElementById('lb-epsilon');
        this.lbMaxScore = document.getElementById('lb-maxscore');
        this.lbAvgScore = document.getElementById('lb-avgscore');

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

        const avgScore = this.telemetry.avgScores.length > 0
            ? this.telemetry.avgScores[this.telemetry.avgScores.length - 1].toFixed(2)
            : "0";
        this.lbAvgScore.innerText = avgScore;

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

        // ── Arcade chart theme ───────────────────────────────
        const pixelFont = "'Press Start 2P', monospace";
        const golden    = '#e8c870';
        const goldenDim = '#6b4820';
        const gridColor = 'rgba(107, 72, 32, 0.35)';

        // Shared axis config
        const axisDefaults = (label) => ({
            display: true,
            title: {
                display: true,
                text: label,
                color: golden,
                font: { family: pixelFont, size: 7 },
                padding: 4,
            },
            ticks: {
                color: '#c8943c',
                font: { family: pixelFont, size: 6 },
                maxTicksLimit: 6,
            },
            grid: {
                color: gridColor,
                lineWidth: 1,
            },
            border: { color: goldenDim },
        });

        // Shared plugin config
        const pluginDefaults = {
            legend: {
                display: true,
                position: 'top',
                labels: {
                    color: '#c8943c',
                    font: { family: pixelFont, size: 7 },
                    boxWidth: 12,
                    boxHeight: 2,
                    padding: 8,
                },
            },
        };

        Chart.defaults.color = '#c8943c';
        Chart.defaults.font.family = pixelFont;

        // ── Reward chart (sky blue) ──────────────────────────
        const ctxReward = document.getElementById('chart-reward').getContext('2d');
        this.chartReward = new Chart(ctxReward, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Reward',
                        data: [],
                        borderColor: 'rgba(121, 195, 244, 0.25)',
                        borderWidth: 1,
                        pointRadius: 0,
                    },
                    {
                        label: 'Moving Avg',
                        data: [],
                        borderColor: '#79c3f4',
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
                    x: { ...axisDefaults('EPISODE') },
                    y: { ...axisDefaults('REWARD'), beginAtZero: false },
                },
                plugins: pluginDefaults,
            }
        });

        // ── Score chart (pipe green) ─────────────────────────
        const ctxScore = document.getElementById('chart-score').getContext('2d');
        this.chartScore = new Chart(ctxScore, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Score',
                        data: [],
                        borderColor: 'rgba(115, 191, 46, 0.25)',
                        borderWidth: 1,
                        pointRadius: 0,
                    },
                    {
                        label: 'Moving Avg',
                        data: [],
                        borderColor: '#73bf2e',
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
                    x: { ...axisDefaults('EPISODE') },
                    y: { ...axisDefaults('SCORE'), beginAtZero: true },
                },
                plugins: pluginDefaults,
            }
        });
    }
}
