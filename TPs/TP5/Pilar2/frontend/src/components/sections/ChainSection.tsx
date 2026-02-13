import { useQuery } from "@tanstack/react-query";
import { fetchChain } from "../../api/chain";
import { SectionCard } from "../SectionCard";

export function ChainSection() {
    const {
        data: blocks,
        // isLoading,
        isFetching,
        refetch,
        error,
    } = useQuery({
        queryKey: ["chain"],
        queryFn: fetchChain,
        select: (data) => data.blocks,
    });

    return (
        <SectionCard
            title="Cadena completa de la blockchain"
            loading={isFetching}
            onReload={refetch}
        >
            {error && <p>Error al cargar la cadena</p>}
            {blocks?.map((block, index) => (
                <div key={block.hash} style={styles.block}>
                    <div style={styles.blockHeader}>
                        block #{index}:
                    </div>

                    {block.blockchain_config && (
                        <div>
                            <br />
                            blockchain_config:
                            <div><b>accepted_algorithm: </b>{block.blockchain_config.accepted_algorithm}</div>
                            <div><b>await_response_duration: </b>{block.blockchain_config.await_response_duration}s</div>
                            <div><b>interval_duration: </b>{block.blockchain_config.interval_duration}s</div>
                            <div><b>max_mining_attempts: </b>{block.blockchain_config.max_mining_attempts}</div>
                            <div><b>prize_amount: </b>{block.blockchain_config.prize_amount}</div>
                            <br />
                        </div>
                    )}
                    <div><b>hash:</b> {block.hash}</div>
                    <div><b>miner_id:</b> {block.miner_id}</div>
                    <div><b>previous_hash:</b> {block.previous_hash}</div>
                    <div><b>nonce:</b> {block.nonce}</div>

                    <div style={styles.txTitle}>transaction:</div>
                    <div style={styles.txFields}><b>source:</b> {block.transaction.source.slice(0, 80)}{block.transaction.source.length > 80 && (<>…</>)}</div>
                    {block.transaction.target && (
                        <div style={styles.txFields}><b>target:</b> {block.transaction.target.slice(0, 80)}{block.transaction.target.length > 80 && (<>…</>)}</div>
                    )}
                    <div style={styles.txFields}><b>amount:</b> {block.transaction.amount}</div>
                    <div style={styles.txFields}><b>timestamp:</b> {block.transaction.timestamp}</div>
                    <div style={styles.txFields}><b>sign:</b> {block.transaction.sign.slice(0, 80)}{block.transaction.sign.length > 80 && (<>…</>)}</div>
                    <hr />
                </div>
            ))}
        </SectionCard>
    )
}

const styles: Record<string, React.CSSProperties> = {
    block: {

    },
    blockHeader: {
        fontSize: "1.2rem"
    },
    txTitle: {
        fontSize: "1.2rem"
    },
    txFields: {
        marginLeft: "1rem"
    },
};