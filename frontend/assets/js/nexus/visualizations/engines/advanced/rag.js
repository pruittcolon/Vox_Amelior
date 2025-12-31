/**
 * RAG Evaluation Visualization
 * Renders metrics for RAG evaluation engine results.
 *
 * @module nexus/visualizations/engines/advanced/rag
 */

import { VIZ_COLORS, formatNumber } from '../../core/viz-utils.js';
import { ensurePlotly } from '../../core/plotly-helpers.js';

export function buildSection(data, vizId) {
    if (!data) return '';
    return `
        <div class="engine-viz-section">
            <h5>RAG Evaluation Metrics</h5>
            <div class="rag-metrics-container" id="rag-metrics-${vizId}"></div>
        </div>
    `;
}

export function render(data, vizId) {
    renderRAGMetrics(data, `rag-metrics-${vizId}`);
}

function renderRAGMetrics(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Check if this is a fallback response (engine couldn't find required columns)
    if (data?.fallback_used || data?.summary?.status === 'fallback_summary') {
        const reason = data?.summary?.reason || 'RAG evaluation requires query, relevant_docs, retrieved_docs, and generated_answer columns.';
        container.innerHTML = `
            <div style="text-align: center; padding: 2rem; color: #94a3b8;">
                <p style="margin-bottom: 0.5rem; font-weight: 500;">Data Requirements Not Met</p>
                <p style="font-size: 0.85rem; color: #64748b;">${reason}</p>
                <p style="font-size: 0.75rem; color: #475569; margin-top: 1rem;">
                    This engine evaluates Retrieval-Augmented Generation (RAG) systems.
                    Upload LLM test data with query-response pairs to see metrics.
                </p>
            </div>
        `;
        return;
    }

    const precision = data?.precision || data?.retrieval_precision || 0;
    const recall = data?.recall || data?.retrieval_recall || 0;
    const f1 = data?.f1_score || data?.f1 || 0;
    const accuracy = data?.accuracy || data?.answer_accuracy || 0;

    container.innerHTML = `
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1rem;">
            <div style="background: ${VIZ_COLORS.surface}; padding: 1.25rem; border-radius: 0.75rem; text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 600; color: ${VIZ_COLORS.primary};">${(precision * 100).toFixed(1)}%</div>
                <div style="color: ${VIZ_COLORS.textMuted}; font-size: 0.8rem;">Precision</div>
            </div>
            <div style="background: ${VIZ_COLORS.surface}; padding: 1.25rem; border-radius: 0.75rem; text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 600; color: ${VIZ_COLORS.accent};">${(recall * 100).toFixed(1)}%</div>
                <div style="color: ${VIZ_COLORS.textMuted}; font-size: 0.8rem;">Recall</div>
            </div>
            <div style="background: ${VIZ_COLORS.surface}; padding: 1.25rem; border-radius: 0.75rem; text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 600; color: ${VIZ_COLORS.success};">${(f1 * 100).toFixed(1)}%</div>
                <div style="color: ${VIZ_COLORS.textMuted}; font-size: 0.8rem;">F1 Score</div>
            </div>
            <div style="background: ${VIZ_COLORS.surface}; padding: 1.25rem; border-radius: 0.75rem; text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 600; color: ${VIZ_COLORS.warning};">${(accuracy * 100).toFixed(1)}%</div>
                <div style="color: ${VIZ_COLORS.textMuted}; font-size: 0.8rem;">Accuracy</div>
            </div>
        </div>
    `;
}

