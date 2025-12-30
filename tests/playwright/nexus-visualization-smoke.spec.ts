import { test, expect } from '@playwright/test';

const fixtures: Record<string, any> = {
  titan: {
    best_model: 'RandomForest',
    accuracy: 0.82,
    target_column: 'read',
    feature_importance: [
      { feature: 'students', stability: 84 },
      { feature: 'teachers', stability: 72 },
      { feature: 'income', stability: 65 },
      { feature: 'english', stability: 58 },
      { feature: 'calworks', stability: 44 },
      { feature: 'expenditure', stability: 39 }
    ],
    variants: [
      { model_name: 'RandomForest', cv_score: 0.82 },
      { model_name: 'GradientBoosting', cv_score: 0.78 },
      { model_name: 'ElasticNet', cv_score: 0.73 }
    ]
  },
  predictive: {
    visualizations: [
      {
        type: 'line_chart_with_forecast',
        title: 'Time Series Forecast',
        data: {
          historical_dates: ['2024-01', '2024-02', '2024-03', '2024-04'],
          historical_values: [100, 110, 105, 120],
          forecast_dates: ['2024-05', '2024-06', '2024-07'],
          forecast_values: [125, 130, 138],
          lower_bound: [115, 118, 125],
          upper_bound: [135, 142, 150]
        }
      },
      {
        type: 'metric_cards',
        title: 'Model Performance',
        data: { MAE: 2.4, RMSE: 3.1 }
      }
    ]
  },
  clustering: {
    n_clusters: 3,
    labels: [0, 0, 1, 1, 2, 2],
    pca_3d: {
      points: [
        { x: -1.2, y: 0.3, z: 0.2, cluster: 0 },
        { x: -0.9, y: 0.5, z: 0.1, cluster: 0 },
        { x: 0.4, y: -0.2, z: 0.3, cluster: 1 },
        { x: 0.6, y: -0.4, z: 0.5, cluster: 1 },
        { x: 1.1, y: 0.8, z: -0.2, cluster: 2 },
        { x: 0.9, y: 0.7, z: -0.1, cluster: 2 }
      ],
      explained_variance: [0.4, 0.22, 0.14],
      total_variance_explained: 0.76,
      component_loadings: [
        {
          component: 'PC1',
          variance_explained: 0.4,
          top_features: [
            { feature: 'students', loading: 0.6 },
            { feature: 'teachers', loading: 0.3 }
          ]
        }
      ]
    },
    cluster_profiles: [
      {
        cluster_id: 0,
        size: 2,
        percentage: 33.3,
        feature_stats: {
          students: { mean: 45, std: 5 },
          teachers: { mean: 12, std: 2 }
        }
      },
      {
        cluster_id: 1,
        size: 2,
        percentage: 33.3,
        feature_stats: {
          students: { mean: 55, std: 6 },
          teachers: { mean: 14, std: 3 }
        }
      }
    ],
    visualizations: [
      {
        type: 'scatter_plot',
        title: 'Cluster Scatter',
        data: {
          x: [10, 12, 14, 16, 18, 20],
          y: [5, 7, 6, 9, 10, 11],
          labels: [0, 0, 1, 1, 2, 2],
          x_label: 'students',
          y_label: 'teachers'
        }
      }
    ]
  },
  anomaly: {
    anomaly_count: 2,
    scores: [0.1, 0.2, 0.9, 0.3, 0.8],
    threshold: 0.7,
    visualizations: [
      {
        type: 'scatter_plot',
        title: 'Anomalies',
        data: {
          x: [1, 2, 3, 4, 5],
          y: [10, 11, 50, 12, 55],
          is_anomaly: [false, false, true, false, true]
        }
      },
      {
        type: 'histogram',
        title: 'Anomaly Score Distribution',
        data: { scores: [0.1, 0.2, 0.9, 0.3, 0.8] }
      }
    ]
  },
  statistical: {
    descriptive: {
      numeric: {
        students: { mean: 50, std: 10, coefficient_of_variation: 0.2 },
        teachers: { mean: 20, std: 4, coefficient_of_variation: 0.2 }
      }
    },
    correlation: {
      pearson: {
        students: { students: 1, teachers: 0.4 },
        teachers: { students: 0.4, teachers: 1 }
      }
    },
    visualizations: [
      {
        type: 'heatmap',
        title: 'Correlation Matrix',
        data: {
          columns: ['students', 'teachers'],
          matrix: {
            students: { students: 1, teachers: 0.4 },
            teachers: { students: 0.4, teachers: 1 }
          }
        }
      }
    ]
  },
  trend: {
    visualizations: [
      {
        type: 'line_chart_with_trend',
        title: 'Trend Analysis',
        data: {
          dates: ['2024-01', '2024-02', '2024-03', '2024-04'],
          values: [100, 110, 115, 130],
          trend_line: [100, 108, 118, 128]
        }
      },
      {
        type: 'text_summary',
        title: 'Change Points',
        data: {
          points: [
            { timestamp: '2024-02', value: 110, magnitude: 10 }
          ]
        }
      }
    ]
  },
  graphs: {
    total_graphs: 12,
    graphs: [
      {
        type: 'grouped_bar_chart',
        title: 'Grouped Pivot',
        data: {
          pivot: {
            SegmentA: { North: 12, South: 18 },
            SegmentB: { North: 9, South: 6 }
          }
        }
      },
      {
        type: 'bar_comparison',
        title: 'Stage Duration: Actual vs Benchmark',
        x_data: ['Stage 1', 'Stage 2', 'Stage 3'],
        series: [
          { name: 'Actual', data: [12, 9, 14], color: '#6366f1' },
          { name: 'Benchmark', data: [10, 11, 12], color: '#94a3b8' }
        ]
      },
      {
        type: 'gauge',
        title: 'Velocity Score',
        value: 68,
        min: 0,
        max: 100,
        thresholds: [
          { value: 50, color: '#ef4444' },
          { value: 75, color: '#f59e0b' },
          { value: 100, color: '#10b981' }
        ]
      },
      {
        type: 'pareto',
        title: 'Pareto Analysis',
        x_data: ['A', 'B', 'C'],
        y_data: [100, 60, 40],
        cumulative: [50, 80, 100]
      },
      {
        type: 'bar_line_combo',
        title: 'Revenue vs Trend',
        x_data: ['Q1', 'Q2', 'Q3'],
        bar_data: [30, 40, 50],
        line_data: [28, 35, 48],
        bar_label: 'Revenue',
        line_label: 'Trend'
      },
      {
        type: 'metric_cards',
        title: 'Key Metrics',
        data: { Avg: 12, Max: 20, Min: 4 }
      },
      {
        type: 'metric_card',
        title: 'Single Metric',
        data: { value: 12, total: 100, percentage: 12, label: 'Utilization' }
      },
      {
        type: 'line_chart_with_markers',
        title: 'Markers',
        data: {
          dates: ['T1', 'T2', 'T3'],
          values: [10, 20, 15],
          is_anomaly: [false, true, false]
        }
      },
      {
        type: 'pie_chart',
        title: 'Segments',
        labels: ['A', 'B'],
        values: [60, 40],
        colors: ['#3b82f6', '#10b981']
      },
      {
        type: 'histogram',
        title: 'Distribution',
        x_data: [1, 2, 2, 3, 4, 4, 5],
        bins: 5
      },
      {
        type: 'scatter_plot',
        title: 'Scatter',
        data: { x: [1, 2, 3], y: [3, 2, 1], labels: ['A', 'B', 'C'] }
      },
      {
        type: 'box_plot',
        title: 'Box',
        data: { values: [10, 12, 14, 9, 11] }
      }
    ]
  },
  cost: {},
  roi: {
    graphs: [
      {
        type: 'forecast',
        title: 'ROI Forecast',
        x_data: ['P1', 'P2', 'P3'],
        y_data: [12, 13, 14],
        lower_bound: [10, 11, 12],
        upper_bound: [14, 15, 16]
      }
    ]
  },
  spend_patterns: {
    graphs: [
      {
        type: 'time_series',
        title: 'Spend Trend',
        x_data: ['2024-01', '2024-02', '2024-03'],
        y_data: [100, 110, 120],
        x_label: 'Period',
        y_label: 'Spend'
      },
      {
        type: 'scatter',
        title: 'Spend Anomalies',
        x_data: [1, 2, 3, 4, 5],
        y_data: [10, 15, 50, 12, 18],
        anomalies: [2]
      }
    ]
  },
  budget_variance: {
    categories: ['Marketing', 'Sales'],
    budgets: [100, 120],
    actuals: [90, 140],
    graphs: [
      {
        type: 'waterfall',
        title: 'Budget to Actual',
        x_data: ['Budget', 'Marketing', 'Sales', 'Actual'],
        y_data: [220, -10, 20, 230],
        measure: ['absolute', 'relative', 'relative', 'absolute']
      },
      {
        type: 'bar_chart_grouped',
        title: 'Budget vs Actual',
        x_data: ['Marketing', 'Sales'],
        datasets: [
          { label: 'Budget', data: [100, 120], color: '#94a3b8' },
          { label: 'Actual', data: [90, 140], color: '#3b82f6' }
        ]
      }
    ]
  },
  profit_margins: {},
  revenue_forecasting: {},
  customer_ltv: {},
  cash_flow: {},
  inventory_optimization: {
    graphs: [
      {
        type: 'bar_chart_grouped',
        title: 'ABC Inventory Analysis',
        x_data: ['A', 'B', 'C'],
        datasets: [
          { label: 'Item Count', data: [5, 8, 12], color: '#94a3b8' },
          { label: 'Value ($)', data: [200, 150, 90], color: '#3b82f6', yAxisID: 'right' }
        ]
      },
      {
        type: 'pie_chart',
        title: 'Stock Status',
        labels: ['OK', 'Reorder'],
        values: [8, 4],
        colors: ['#10b981', '#ef4444']
      }
    ]
  },
  pricing_strategy: {},
  market_basket: {
    rules: [
      {
        antecedent: ['A'],
        consequent: ['B'],
        support: 0.2,
        confidence: 0.6,
        lift: 1.4
      },
      {
        antecedent: ['B'],
        consequent: ['C'],
        support: 0.15,
        confidence: 0.5,
        lift: 1.2
      }
    ]
  },
  resource_utilization: {
    graphs: [
      {
        type: 'bar_chart',
        title: 'Resource Utilization',
        x_data: ['R1', 'R2', 'R3'],
        y_data: [70, 90, 55],
        colors: ['#3b82f6', '#ef4444', '#10b981']
      }
    ]
  },
  rag_evaluation: {
    visualizations: [
      {
        type: 'gauge_chart',
        title: 'MRR',
        data: { value: 0.72, max: 1.0 }
      },
      {
        type: 'radar_chart',
        title: 'Generation Quality',
        data: {
          Faithfulness: 80,
          Relevance: 75,
          'No Hallucination': 90
        }
      }
    ]
  },
  chaos: {
    complexity_matrix: {
      columns: ['a', 'b', 'c'],
      dcor: {
        a: { a: 1, b: 0.3, c: 0.2 },
        b: { a: 0.3, b: 1, c: 0.5 },
        c: { a: 0.2, b: 0.5, c: 1 }
      }
    }
  },
  oracle: {
    feature_importance: [
      { name: 'income', importance: 0.4 },
      { name: 'teachers', importance: 0.3 }
    ]
  }
};

const defaultFixture = {
  summary: { row_count: 50 },
  graphs: [
    {
      type: 'bar_chart',
      title: 'Default Graph',
      labels: ['A', 'B'],
      values: [1, 2]
    }
  ]
};

test('nexus visualizations render across all engines', async ({ page }) => {
  const consoleErrors: string[] = [];

  page.on('pageerror', error => {
    consoleErrors.push(error.message);
  });

  page.on('console', message => {
    if (message.type() === 'error') {
      const location = message.location();
      const locator = location?.url ? ` @ ${location.url}:${location.lineNumber || 0}` : '';
      consoleErrors.push(`${message.text()}${locator}`);
    }
  });

  await page.route('**/upload', route => {
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        filename: 'sample.csv',
        columns: ['students', 'teachers', 'income', 'read'],
        row_count: 50
      })
    });
  });

  await page.route('**/api/ml/gpu-status', route => {
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ cuda_available: false, devices: [] })
    });
  });

  await page.route('**/api/gpu-coordinator/gpu/state', route => {
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ available: true, owner: 'nexus' })
    });
  });

  await page.route('**/api/public/chat', route => {
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ message: 'Summary ready.' })
    });
  });

  await page.route('**/analytics/run-engine/*', route => {
    const url = route.request().url();
    const engineName = url.split('/').pop() || '';
    const body = fixtures[engineName] || defaultFixture;
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(body)
    });
  });

  await page.goto('/nexus.html');

  await page.setInputFiles('#file-input', {
    name: 'sample.csv',
    mimeType: 'text/csv',
    buffer: Buffer.from('students,teachers,income,read\n1,1,1,1')
  });

  const analyzeButton = page.locator('#analyze-btn');
  await expect(analyzeButton).toBeEnabled();
  await analyzeButton.click();

  await page.waitForFunction(() => {
    const status = document.getElementById('engines-status');
    return status && status.textContent && status.textContent.includes('Complete');
  }, { timeout: 60000 });

  const cards = page.locator('#all-engines-results .engine-result-card');
  await expect(cards).toHaveCount(22);

  const plotlyCharts = page.locator('#all-engines-results .js-plotly-plot');
  const plotlyCount = await plotlyCharts.count();
  expect(plotlyCount).toBeGreaterThan(0);

  const clusterCard = page.locator('#all-engines-results .engine-result-card[data-engine="clustering"]');
  await expect(clusterCard).not.toContainText('No cluster data available');

  await expect(page.locator('#all-engines-results .engine-result-card[data-engine="titan"] .waterfall-container .js-plotly-plot')).toHaveCount(1);
  await expect(page.locator('#all-engines-results .engine-result-card[data-engine="clustering"] .scatter-3d-container .js-plotly-plot')).toHaveCount(1);
  await expect(page.locator('#all-engines-results .engine-result-card[data-engine="clustering"] .cluster-card')).toHaveCount(2);
  await expect(page.locator('#all-engines-results .engine-result-card[data-engine="clustering"] .pca-explanation')).toBeVisible();
  await expect(page.locator('#all-engines-results .engine-result-card[data-engine="anomaly"] .anomaly-dist-container .js-plotly-plot')).toHaveCount(1);
  await expect(page.locator('#all-engines-results .engine-result-card[data-engine="statistical"] .boxplot-container .js-plotly-plot')).toHaveCount(1);

  await page.waitForFunction((engines) => {
    return engines.every((engine) => {
      const card = document.querySelector(`#all-engines-results .engine-result-card[data-engine="${engine}"]`);
      if (!card) return false;
      const containers = card.querySelectorAll('.fin-chart-container, .fin-chart-container-lg, .fin-chart-container-xl');
      if (!containers.length) return false;
      return Array.from(containers).every(container => container.children.length > 0);
    });
  }, { timeout: 60000 }, [
    'cash_flow',
    'budget_variance',
    'profit_margins',
    'revenue_forecasting',
    'customer_ltv',
    'cost',
    'roi',
    'market_basket',
    'pricing_strategy',
    'spend_patterns',
    'inventory_optimization',
    'resource_utilization'
  ]);

  await expect(page.locator('#all-engines-results').getByText('Additional Visualizations')).toHaveCount(0);

  expect(consoleErrors).toEqual([]);
});
