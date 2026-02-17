export type TransactionView = {
    source: string;
    target: string;
    amount: number;
    timestamp: string;
    sign: string;
};

export type ActiveTransactionView = {
    transaction: TransactionView;
    ttl: number;
};

export type MinedBlockView = {
    previous_hash: string;
    nonce: number;
    miner_id: string;
    hash: string;
    transaction: TransactionView;
};

export type LastCycleSummary = {
    completed_at: string;
    received_chains_count: number;
    selected_blocks: MinedBlockView[];
    active_before_cycle: ActiveTransactionView[];
    requeued_transactions: ActiveTransactionView[];
    discarded_transactions: ActiveTransactionView[];
    current_active_transactions: ActiveTransactionView[];
    counts: {
        active_before_cycle: number;
        mined_blocks: number;
        requeued_transactions: number;
        discarded_transactions: number;
        current_active_transactions: number;
    };
};

export type CycleSummaryResponse = {
    last_cycle: LastCycleSummary | null;
    current: {
        state: string;
        counts: {
            active_transactions: number;
            pending_transactions: number;
        };
        active_transactions: ActiveTransactionView[];
        pending_transactions: ActiveTransactionView[];
    };
};

export async function fetchCycleSummary(): Promise<CycleSummaryResponse> {
    const res = await fetch("http://localhost:8000/cycle/summary");
    if (!res.ok) {
        throw new Error("Failed to fetch cycle summary");
    }
    return res.json() as Promise<CycleSummaryResponse>;
}
