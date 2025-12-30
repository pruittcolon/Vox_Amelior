/**
 * Nexus Financial Theme Overrides
 * Light-theme friendly palette for financial chart modules.
 */

(function () {
    const theme = {
        colors: {
            primary: '#02559e',
            secondary: '#2b7fd4',
            success: '#10b981',
            danger: '#ef4444',
            warning: '#f59e0b',
            info: '#0ea5e9',
            neutral: '#64748b',
            revenue: '#16a34a',
            cost: '#ef4444',
            profit: '#10b981',
            loss: '#dc2626',
            forecast: '#2563eb',
            actual: '#0ea5e9',
            budget: '#94a3b8',
            variance: '#f97316',
            gradientStart: '#2b7fd4',
            gradientEnd: '#67e8f9'
        },
        palettes: {
            financial: ['#02559e', '#2b7fd4', '#0ea5e9', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'],
            sequential: ['#f1f5f9', '#cbd5e1', '#94a3b8', '#64748b', '#475569'],
            diverging: ['#ef4444', '#fca5a5', '#e2e8f0', '#86efac', '#16a34a'],
            categorical: ['#0ea5e9', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#14b8a6']
        },
        fonts: {
            primary: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
            mono: '"JetBrains Mono", "Fira Code", monospace',
            sizes: {
                xs: 10,
                sm: 11,
                md: 12,
                lg: 14,
                xl: 16,
                xxl: 20,
                hero: 28
            }
        },
        backgrounds: {
            chart: 'rgba(255,255,255,0)',
            card: 'rgba(248,250,252,0.9)',
            hover: 'rgba(2,85,158,0.08)',
            grid: 'rgba(148,163,184,0.2)'
        },
        borders: {
            color: 'rgba(148,163,184,0.4)',
            radius: 12,
            width: 1
        },
        shadows: {
            sm: '0 1px 2px rgba(15, 23, 42, 0.08)',
            md: '0 6px 16px rgba(15, 23, 42, 0.12)',
            lg: '0 12px 24px rgba(15, 23, 42, 0.18)',
            glow: '0 0 18px rgba(2, 85, 158, 0.18)'
        },
        animations: {
            fast: 200,
            normal: 400,
            slow: 800,
            reveal: 1200
        }
    };

    window.FinancialTheme = theme;

    window.PlotlyBaseConfig = {
        displayModeBar: false,
        responsive: true,
        scrollZoom: false,
        staticPlot: false,
        locale: 'en-US'
    };

    window.PlotlyBaseLayout = {
        paper_bgcolor: 'rgba(255,255,255,0)',
        plot_bgcolor: 'rgba(255,255,255,0)',
        font: {
            family: theme.fonts.primary,
            size: theme.fonts.sizes.md,
            color: '#1e293b'
        },
        margin: { l: 60, r: 30, t: 40, b: 60, pad: 4 },
        xaxis: {
            gridcolor: theme.backgrounds.grid,
            linecolor: theme.borders.color,
            zerolinecolor: theme.borders.color,
            tickfont: { color: '#64748b', size: 11 },
            titlefont: { color: '#1e293b', size: 12 }
        },
        yaxis: {
            gridcolor: theme.backgrounds.grid,
            linecolor: theme.borders.color,
            zerolinecolor: theme.borders.color,
            tickfont: { color: '#64748b', size: 11 },
            titlefont: { color: '#1e293b', size: 12 }
        },
        legend: {
            font: { color: '#1e293b', size: 11 },
            bgcolor: 'rgba(255,255,255,0)',
            bordercolor: 'rgba(255,255,255,0)'
        },
        hoverlabel: {
            bgcolor: '#ffffff',
            bordercolor: theme.borders.color,
            font: { family: theme.fonts.primary, color: '#1e293b', size: 12 }
        }
    };

    window.EChartsBaseTheme = {
        backgroundColor: 'transparent',
        textStyle: {
            fontFamily: theme.fonts.primary,
            color: '#1e293b'
        },
        title: {
            textStyle: {
                color: '#0f172a',
                fontWeight: 600,
                fontSize: 16
            },
            subtextStyle: {
                color: '#64748b',
                fontSize: 12
            }
        },
        legend: {
            textStyle: {
                color: '#1e293b'
            }
        },
        tooltip: {
            backgroundColor: '#ffffff',
            borderColor: theme.borders.color,
            borderWidth: 1,
            textStyle: {
                color: '#1e293b'
            },
            extraCssText: 'box-shadow: 0 4px 12px rgba(15, 23, 42, 0.12); border-radius: 8px;'
        },
        color: theme.palettes.financial
    };
})();
