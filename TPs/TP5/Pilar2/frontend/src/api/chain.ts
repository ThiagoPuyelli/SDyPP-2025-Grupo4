export async function fetchChain(): Promise<BlockchainState> {
    const res = await fetch("api/chain")
    if (!res.ok) {
        throw new Error("Failed to fetch chain");
    }
    return res.json()
}

export interface Transaction {
    source: string;
    target?: string;
    amount: number;
    timestamp: string;
    sign: string;
}

export interface Block {
    previous_hash: string;
    transaction: Transaction;
    nonce: number;
    miner_id: string;
    hash: string;
    blockchain_config?: {
        interval_duration: number;
        await_response_duration: number;
        max_mining_attempts: number;
        accepted_algorithm: string;
        prize_amount: number;
    };
}

export interface BlockchainState {
    blocks: Block[];
}