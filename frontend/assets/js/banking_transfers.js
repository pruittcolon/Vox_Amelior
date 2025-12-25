/**
 * Banking Transfers Module
 * Handles internal transfers, bill pay, and P2P logic
 */

async function executeTransfer() {
    const fromAccount = document.getElementById('transferFrom').value;
    const toAccount = document.getElementById('transferTo').value;
    const amount = parseFloat(document.getElementById('transferAmount').value);
    const memo = document.getElementById('transferMemo').value;

    if (!fromAccount || !toAccount) {
        alert('Please select both source and destination accounts.');
        return;
    }
    if (!amount || amount <= 0) {
        alert('Please enter a valid amount.');
        return;
    }

    const verifyMsg = `Confirm transfer of $${amount.toFixed(2)}?\nFrom: ${fromAccount}\nTo: ${toAccount}`;
    if (!confirm(verifyMsg)) return;

    // Show loading state
    const btn = document.querySelector('button[onclick="executeTransfer()"]');
    const originalText = btn.textContent;
    btn.textContent = 'Processing...';
    btn.disabled = true;

    try {
        // In fully implemented backend, this would hit /fiserv/api/v1/transfer
        // For Phase 1, we might simulate or hit a mock endpoint
        const payload = {
            from_account_id: fromAccount,
            to_account_id: toAccount,
            amount: amount,
            description: memo || 'Transfer'
        };

        const resp = await fetch('/fiserv/api/v1/transfer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await resp.json();

        if (data.success) {
            alert(`Success! Transaction ID: ${data.transaction_id}`);
            // Reset form
            document.getElementById('transferAmount').value = '';
            document.getElementById('transferMemo').value = '';
            // Refresh account list if visible
            if (typeof refreshAccountList === 'function') refreshAccountList();
        } else {
            alert('Transfer failed: ' + (data.error || 'Unknown error'));
        }

    } catch (e) {
        console.error('Transfer error:', e);
        alert('System Error: ' + e.message);
    } finally {
        btn.textContent = originalText;
        btn.disabled = false;
    }
}
