import { useQuery } from "@tanstack/react-query";
import { fetchCycleSummary } from "../../api/cycleSummary";
import { SectionCard } from "../SectionCard";

function shorten(value: string, start = 10, end = 8): string {
    if (value.length <= start + end + 3) {
        return value;
    }
    return `${value.slice(0, start)}...${value.slice(-end)}`;
}

export function CycleSummarySection() {
    const { data, isFetching, refetch, error } = useQuery({
        queryKey: ["cycle-summary"],
        queryFn: fetchCycleSummary,
        refetchInterval: 5000,
    });

    const lastCycle = data?.last_cycle;
    const current = data?.current;

    return (
        <SectionCard
            title="Ultimo ciclo y pendientes"
            loading={isFetching}
            onReload={() => void refetch()}
        >
            {error && <p>Error al cargar resumen del ciclo</p>}

            {!error && !data && <p>Sin datos disponibles.</p>}

            {current && (
                <>
                    <p>
                        Estado actual: <strong>{current.state}</strong>
                    </p>
                    <p>
                        Pendientes a procesar (activas):{" "}
                        <strong>{current.counts.active_transactions}</strong>
                    </p>
                    <p>
                        En cola pendiente:{" "}
                        <strong>{current.counts.pending_transactions}</strong>
                    </p>
                </>
            )}

            {lastCycle ? (
                <>
                    <hr style={styles.separator} />
                    <p>
                        Ultimo cierre: <strong>{lastCycle.completed_at}</strong>
                    </p>
                    <p>
                        Bloques procesados en ultimo ciclo:{" "}
                        <strong>{lastCycle.counts.mined_blocks}</strong>
                    </p>
                    <p>
                        Reencoladas: <strong>{lastCycle.counts.requeued_transactions}</strong>
                        {" | "}
                        Descartadas: <strong>{lastCycle.counts.discarded_transactions}</strong>
                    </p>

                    <p style={styles.subtitle}>Bloques procesados (ultimo ciclo)</p>
                    {lastCycle.selected_blocks.length === 0 ? (
                        <p style={styles.empty}>Sin bloques minados en el ultimo ciclo.</p>
                    ) : (
                        <ul style={styles.list}>
                            {lastCycle.selected_blocks.map((block) => (
                                <li key={block.hash} style={styles.listItem}>
                                    <strong>hash:</strong> {shorten(block.hash)}{" "}
                                    <strong>miner:</strong> {shorten(block.miner_id, 12, 10)}{" "}
                                    <strong>monto:</strong> {block.transaction.amount}
                                </li>
                            ))}
                        </ul>
                    )}

                    <p style={styles.subtitle}>Pendientes actuales (transacciones activas)</p>
                    {current && current.active_transactions.length === 0 ? (
                        <p style={styles.empty}>No hay transacciones activas.</p>
                    ) : (
                        <ul style={styles.list}>
                            {current?.active_transactions.map((activeTx) => (
                                <li key={activeTx.transaction.sign} style={styles.listItem}>
                                    <strong>sign:</strong> {shorten(activeTx.transaction.sign)}{" "}
                                    <strong>ttl:</strong> {activeTx.ttl}{" "}
                                    <strong>monto:</strong> {activeTx.transaction.amount}
                                </li>
                            ))}
                        </ul>
                    )}
                </>
            ) : (
                <p style={styles.empty}>
                    Aun no hay resumen del ultimo ciclo. Se genera al cerrar OPEN_TO_RESULTS.
                </p>
            )}
        </SectionCard>
    );
}

const styles: Record<string, React.CSSProperties> = {
    separator: {
        border: 0,
        borderTop: "1px solid #e5e7eb",
        margin: "12px 0",
    },
    subtitle: {
        fontWeight: 600,
        marginTop: 8,
        marginBottom: 6,
    },
    list: {
        margin: 0,
        paddingLeft: 18,
    },
    listItem: {
        marginBottom: 4,
        wordBreak: "break-word",
    },
    empty: {
        color: "#6b7280",
    },
};
