/**
 * Salesforce Real-Time Streaming Client
 * Handles WebSocket connection and updates UI elements based on CDC events.
 */

class SalesforceStream {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/v1/salesforce/stream`;

        console.log(`[Salesforce] Connecting to stream: ${wsUrl}`);
        this.socket = new WebSocket(wsUrl);

        this.socket.onopen = () => {
            console.log('[Salesforce] Stream connected');
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.updateStatus(true);
            this.pulseHeader();
        };

        this.socket.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleMessage(message);
            } catch (e) {
                console.error('[Salesforce] Error parsing message:', e);
            }
        };

        this.socket.onclose = () => {
            console.log('[Salesforce] Stream disconnected');
            this.isConnected = false;
            this.updateStatus(false);
            this.attemptReconnect();
        };

        this.socket.onerror = (error) => {
            console.error('[Salesforce] Stream error:', error);
        };
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 10000);
            console.log(`[Salesforce] Reconnecting in ${delay}ms...`);
            setTimeout(() => this.connect(), delay);
        }
    }

    updateStatus(connected) {
        const statusDot = document.getElementById('sync-dot');
        const statusText = document.getElementById('sync-text');

        if (statusDot && statusText) {
            if (connected) {
                statusDot.style.backgroundColor = '#4ade80'; // Green
                statusText.innerText = 'Live Updates';
                statusDot.classList.add('pulse');
            } else {
                statusDot.style.backgroundColor = '#ef4444'; // Red
                statusText.innerText = 'Offline';
                statusDot.classList.remove('pulse');
            }
        }
    }

    pulseHeader() {
        const header = document.querySelector('.sf-header');
        if (header) {
            header.classList.add('flash-success');
            setTimeout(() => header.classList.remove('flash-success'), 1000);
        }
    }

    handleMessage(message) {
        console.log('[Salesforce] Event received:', message);
        const { type, data } = message;

        // Visual feedback
        this.pulseHeader();

        // Update Recent Activity
        if (data && data.entity_name && data.change_type) {
            this.addActivityItem(data);
        }

        // Update KPIs based on event type
        if (type === 'opportunity_update') {
            this.updateOpportunityKPIs(data);
        }
    }

    addActivityItem(data) {
        const activityList = document.querySelector('.sf-activity-list');
        if (!activityList) return;

        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const icon = this.getIconForEntity(data.entity_name);

        const itemHtml = `
            <div class="sf-activity-item new-item">
                <div class="sf-activity-icon">${icon}</div>
                <div class="sf-activity-content">
                    <div class="sf-activity-title">${data.entity_name} ${data.change_type}</div>
                    <div class="sf-activity-desc">${data.record_name || 'Record'} updated via CDC</div>
                    <div class="sf-activity-time">${time}</div>
                </div>
            </div>
        `;

        activityList.insertAdjacentHTML('afterbegin', itemHtml);

        // Remove "new-item" class after animation
        setTimeout(() => {
            const newItem = activityList.querySelector('.new-item');
            if (newItem) newItem.classList.remove('new-item');
        }, 3000);
    }

    getIconForEntity(entity) {
        const map = {
            'Opportunity': '💰',
            'Account': 'Gd',
            'Contact': '👤',
            'Lead': '⚡'
        };
        return map[entity] || '📝';
    }

    updateOpportunityKPIs(data) {
        // In a full implementation, we would accumulate changes or fetch fresh stats.
        // For visual feedback now, we'll just flash the Revenue card.
        const revCard = document.getElementById('kpi-revenue');
        if (revCard) {
            revCard.parentElement.classList.add('flash-update');
            setTimeout(() => revCard.parentElement.classList.remove('flash-update'), 1000);
        }
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.salesforceStream = new SalesforceStream();
    window.salesforceStream.connect();
});
